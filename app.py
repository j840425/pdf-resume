"""Streamlit application for PDF RAG system."""
import streamlit as st
import logging
from pathlib import Path
import tempfile
import json

from config import settings, DetailLevel
from services.vertex_service import VertexService
from services.storage_service import StorageService
from core.pdf_processor import PDFProcessor
from core.structure_detector import StructureDetector
from core.embeddings_manager import EmbeddingsManager
from core.summary_generator import SummaryGenerator
from core.qa_engine import QAEngine
from utils.chunking import TextChunker
from utils.validators import PDFValidator, QueryValidator
from utils.cache import DocumentCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📄",
    layout="wide"
)


def initialize_services():
    """Initialize all services."""
    if 'services_initialized' not in st.session_state:
        logger.info("Initializing services...")

        # Initialize Vertex AI service
        st.session_state.vertex_service = VertexService(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model_name=settings.vertex_ai_model,
            embedding_model_name=settings.embedding_model
        )
        st.session_state.vertex_service.initialize()

        # Initialize storage service
        st.session_state.storage_service = StorageService(
            storage_type=settings.storage_type,
            local_path=settings.local_pdf_path,
            gcs_bucket_name=settings.gcs_bucket_name
        )

        # Initialize processors
        st.session_state.pdf_processor = PDFProcessor(max_pages=settings.max_pdf_pages)
        st.session_state.structure_detector = StructureDetector(
            st.session_state.vertex_service
        )
        st.session_state.embeddings_manager = EmbeddingsManager(
            st.session_state.vertex_service,
            settings.embedding_model
        )
        st.session_state.summary_generator = SummaryGenerator(
            st.session_state.vertex_service
        )

        # Initialize validators
        st.session_state.pdf_validator = PDFValidator(
            max_file_size_mb=100,
            max_pages=settings.max_pdf_pages
        )
        st.session_state.query_validator = QueryValidator()

        # Initialize text chunker
        st.session_state.text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # Initialize document cache
        st.session_state.document_cache = DocumentCache()

        st.session_state.services_initialized = True
        logger.info("Services initialized successfully")


def process_pdf(uploaded_file, detail_level: DetailLevel):
    """Process uploaded PDF file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = Path(tmp_file.name)

        logger.info(f"Processing PDF: {uploaded_file.name}")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Validate and extract metadata
        status_text.text("Validando PDF...")
        validation = st.session_state.pdf_processor.validate_pdf(tmp_path)

        if not validation['valid']:
            st.error(f"Error: {validation['error']}")
            return None

        progress_bar.progress(10)

        # Step 2: Extract text
        status_text.text("Extrayendo texto...")
        pages_text = st.session_state.pdf_processor.extract_text_by_pages(tmp_path)
        progress_bar.progress(25)

        # Step 3: Extract TOC and detect structure
        status_text.text("Detectando estructura del documento (analizando índice con LLM)...")
        toc = st.session_state.pdf_processor.extract_table_of_contents(tmp_path)

        # Get first few pages for structure detection
        sample_text = "\n\n".join([pages_text.get(i, "") for i in range(min(5, len(pages_text)))])

        # Detect structure with full page analysis
        structure = st.session_state.structure_detector.detect_structure(
            sample_text,
            toc,
            validation['num_pages'],
            pages_text  # Pass full pages for TOC detection
        )

        # Get summary units (level 2 subsections, or level 1 if no level 2)
        summary_sections = st.session_state.structure_detector.get_summary_units(structure)

        # Get hierarchical structure for display
        hierarchical_structure = st.session_state.structure_detector.get_hierarchical_structure(structure)

        progress_bar.progress(40)

        # Step 4: Create chunks (use ALL sections for chunking, not just summary units)
        status_text.text("Creando chunks del documento...")
        chunks = st.session_state.text_chunker.chunk_by_sections(
            structure['sections'],
            pages_text
        )
        progress_bar.progress(55)

        # Step 5: Generate embeddings
        status_text.text("Generando embeddings...")
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = st.session_state.embeddings_manager.generate_batch_embeddings(
            chunk_texts,
            batch_size=50
        )

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        progress_bar.progress(70)

        # Step 6: Generate summaries (ONLY for summary units - level 2 or level 1)
        status_text.text(f"Generando resúmenes (nivel de subsecciones)...")
        sections_with_content = []

        for section in summary_sections:
            # Collect content for section
            section_content = []
            for page_num in range(section['start_page'], section['end_page']):
                if page_num in pages_text:
                    section_content.append(pages_text[page_num])

            sections_with_content.append({
                'section': section,
                'content': "\n\n".join(section_content)
            })

        # Generate summaries
        summaries = st.session_state.summary_generator.generate_batch_summaries(
            sections_with_content,
            detail_level
        )
        progress_bar.progress(90)

        # Step 7: Generate overall summary
        status_text.text("Generando resumen general...")
        overall_summary = st.session_state.summary_generator.generate_document_summary(
            summaries,
            detail_level
        )

        progress_bar.progress(100)
        status_text.text("¡Procesamiento completado!")

        # Store results in session state
        result = {
            'file_name': uploaded_file.name,
            'metadata': validation,
            'structure': structure,
            'hierarchical_structure': hierarchical_structure,
            'summary_sections': summary_sections,
            'summaries': summaries,
            'overall_summary': overall_summary,
            'chunks': chunks,
            'pages_text': pages_text
        }

        # Clean up temp file
        tmp_path.unlink()

        # Save to cache
        st.session_state.document_cache.save_document(result)

        return result

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error al procesar el PDF: {str(e)}")
        return None


def main():
    """Main application."""
    st.title("📄 Sistema RAG para Análisis de PDFs")
    st.markdown("Carga un PDF de hasta 2000 páginas y obtén resúmenes automáticos organizados por secciones.")

    # Initialize services
    initialize_services()

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuración")

    detail_level = st.sidebar.selectbox(
        "Nivel de detalle:",
        options=[DetailLevel.EXECUTIVE, DetailLevel.NORMAL, DetailLevel.DETAILED],
        format_func=lambda x: {
            DetailLevel.EXECUTIVE: "Ejecutivo (conciso)",
            DetailLevel.NORMAL: "Normal (balanceado)",
            DetailLevel.DETAILED: "Detallado (exhaustivo)"
        }[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Entorno:** {settings.environment.value}")
    st.sidebar.markdown(f"**Almacenamiento:** {settings.storage_type.value}")

    # Main content
    tab1, tab2 = st.tabs(["📤 Cargar Documento", "💬 Preguntas y Respuestas"])

    with tab1:
        st.header("Cargar y Procesar PDF")

        # Check if there's a cached document
        cached_info = st.session_state.document_cache.get_cached_document_info()

        if cached_info:
            st.info(f"📂 Último documento procesado: **{cached_info['file_name']}** "
                   f"({cached_info['num_pages']} páginas, {cached_info['num_sections']} secciones)")

            col_load, col_clear = st.columns([1, 1])

            with col_load:
                if st.button("📥 Cargar Último Documento", type="primary"):
                    with st.spinner("Cargando documento desde caché..."):
                        cached_doc = st.session_state.document_cache.load_document()
                        if cached_doc:
                            st.session_state.current_document = cached_doc
                            st.success("✓ Documento cargado desde caché")
                            st.rerun()
                        else:
                            st.error("Error al cargar el documento desde caché")

            with col_clear:
                if st.button("🗑️ Limpiar Caché"):
                    st.session_state.document_cache.clear_cache()
                    if 'current_document' in st.session_state:
                        del st.session_state.current_document
                    st.success("✓ Caché limpiado")
                    st.rerun()

            st.markdown("---")

        uploaded_file = st.file_uploader(
            "Selecciona un archivo PDF",
            type=['pdf'],
            help=f"Máximo {settings.max_pdf_pages} páginas"
        )

        if uploaded_file:
            # Validate file
            validation = st.session_state.pdf_validator.validate_uploaded_file(uploaded_file)

            if not validation['valid']:
                for error in validation['errors']:
                    st.error(error)
            else:
                st.success(f"✓ Archivo válido ({validation['file_size_mb']:.1f} MB)")

                if st.button("🚀 Procesar Documento", type="primary"):
                    with st.spinner("Procesando documento..."):
                        result = process_pdf(uploaded_file, detail_level)

                        if result:
                            st.session_state.current_document = result
                            st.success("¡Documento procesado exitosamente!")

        # Display results if available
        if 'current_document' in st.session_state:
            st.markdown("---")
            st.header("📊 Resultados")

            doc = st.session_state.current_document

            # Document info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Páginas", doc['metadata']['num_pages'])
            with col2:
                st.metric("Secciones", len(doc['summaries']))
            with col3:
                st.metric("Chunks", len(doc['chunks']))

            # Overall summary
            st.subheader("📝 Resumen General")
            st.markdown(doc['overall_summary'])

            # Section summaries - flat list
            st.subheader("📑 Resúmenes por Sección")

            # Display all summaries as a flat list
            for summary_item in doc['summaries']:
                section = summary_item['section']
                summary = summary_item['summary']

                # Clean title - remove CHAPTER prefix and section numbers
                title = section['title']
                title = title.replace('CHAPTER', '').strip()
                # Remove leading numbers like "1 ", "2 ", etc.
                import re
                title = re.sub(r'^\d+\s+', '', title)

                with st.expander(
                    f"**{title}** (Páginas {section['start_page']}-{section['end_page']})"
                ):
                    st.markdown(summary)

                st.markdown("---")

            # Download results
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                # Download summaries as JSON
                summaries_json = json.dumps(doc['summaries'], indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Descargar Resúmenes (JSON)",
                    data=summaries_json,
                    file_name=f"{doc['file_name']}_summaries.json",
                    mime="application/json"
                )

            with col2:
                # Download overall summary as text
                st.download_button(
                    "📥 Descargar Resumen General (TXT)",
                    data=doc['overall_summary'],
                    file_name=f"{doc['file_name']}_summary.txt",
                    mime="text/plain"
                )

    with tab2:
        st.header("Preguntas sobre el Documento")

        if 'current_document' not in st.session_state:
            st.info("⬅️ Primero carga y procesa un documento en la pestaña 'Cargar Documento'")
        else:
            # Initialize Q&A engine if not already done
            if 'qa_engine' not in st.session_state:
                st.session_state.qa_engine = QAEngine(
                    st.session_state.vertex_service,
                    st.session_state.embeddings_manager,
                    None  # Vector search not required for local mode
                )

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])

                    if message['role'] == 'assistant' and 'sources' in message:
                        with st.expander("📚 Fuentes"):
                            for source in message['sources']:
                                st.markdown(
                                    f"- **{source['section']}** (Página {source['page']}) - "
                                    f"Relevancia: {source['relevance']:.2f}"
                                )

            # User input
            user_question = st.chat_input("Haz una pregunta sobre el documento...")

            if user_question:
                # Validate query
                validation = st.session_state.query_validator.validate_query(user_question)

                if not validation['valid']:
                    st.error(validation['errors'][0])
                else:
                    # Add user message to chat
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })

                    with st.chat_message("user"):
                        st.markdown(user_question)

                    # Generate answer
                    with st.chat_message("assistant"):
                        with st.spinner("Generando respuesta..."):
                            result = st.session_state.qa_engine.answer_question(
                                user_question,
                                st.session_state.current_document['chunks'],
                                top_k=5,
                                summaries=st.session_state.current_document.get('summaries', []),
                                structure=st.session_state.current_document.get('structure', {})
                            )

                            st.markdown(result['answer'])

                            if result['sources']:
                                with st.expander("📚 Fuentes"):
                                    for source in result['sources']:
                                        st.markdown(
                                            f"- **{source['section']}** (Página {source['page']}) - "
                                            f"Relevancia: {source['relevance']:.2f}"
                                        )

                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['answer'],
                        'sources': result['sources']
                    })

                    st.rerun()


if __name__ == "__main__":
    main()
