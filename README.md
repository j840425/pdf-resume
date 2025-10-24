# Sistema RAG para Análisis de PDFs

Sistema de Retrieval-Augmented Generation (RAG) que permite cargar documentos PDF de hasta 2000 páginas y obtener resúmenes automáticos organizados por secciones, además de realizar preguntas sobre el contenido utilizando Vertex AI.

## Características Principales

- **Procesamiento de PDFs grandes**: Hasta 2000 páginas con extracción optimizada de texto
- **Detección automática de estructura**: Identifica tabla de contenidos y secciones mediante análisis con IA (Gemini 2.5 Pro)
- **Resúmenes multinivel**:
  - Ejecutivo: 2-3 párrafos con puntos críticos
  - Normal: 3-5 párrafos con balance entre detalle y concisión
  - Detallado: 5-8+ párrafos exhaustivos con datos técnicos
- **Sistema Q&A inteligente**: Respuestas contextuales con referencias a secciones específicas
- **Deployment flexible**: Desarrollo local y producción en Google Cloud Run
- **Almacenamiento configurable**: Local (desarrollo) o Google Cloud Storage (producción)
- **Cache inteligente**: Caché de documentos procesados para mejorar rendimiento

## Arquitectura

### Componentes Principales

```
┌─────────────────────────────────────────────────────┐
│              Capa de Presentación                   │
│                   (Streamlit)                       │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              Capa de Procesamiento                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ PDF Processor│  │   Structure  │  │ Embeddings│ │
│  │              │→ │   Detector   │→ │  Manager  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Summary    │  │  QA Engine   │  │  Chunking │ │
│  │  Generator   │  │              │  │  Utils    │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              Capa de Servicios                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │   Vertex AI  │  │   Storage    │                │
│  │   Service    │  │   Service    │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              Capa de Datos                          │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ Local/GCS    │  │  Vertex AI   │                │
│  │  Storage     │  │  Embeddings  │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

### Stack Tecnológico

- **Framework**: Streamlit 1.28+
- **IA y ML**:
  - Vertex AI (Gemini 2.5 Pro para generación y análisis)
  - Text Embeddings 004 para vectorización
- **Cloud**: Google Cloud Platform (Cloud Run, Cloud Storage, Vertex AI)
- **PDF Processing**: PyPDF2, pdfplumber
- **Validación**: Pydantic, Pydantic-settings
- **Deployment**: Docker, Cloud Build, Cloud Run

## Estructura del Proyecto

```
pry_pdf_resume/
├── app.py                      # Aplicación Streamlit principal
├── config.py                   # Configuración con Pydantic Settings
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Imagen Docker para Cloud Run
├── .dockerignore              # Archivos excluidos del build
├── .env.development           # Variables de entorno (desarrollo)
├── .env.production            # Variables de entorno (producción)
├── .gitignore                 # Archivos excluidos de Git
│
├── core/                      # Módulos principales del sistema
│   ├── pdf_processor.py       # Extracción y análisis de PDFs
│   ├── structure_detector.py  # Detección automática de estructura
│   ├── embeddings_manager.py  # Generación y gestión de embeddings
│   ├── summary_generator.py   # Generación de resúmenes multinivel
│   └── qa_engine.py           # Motor de preguntas y respuestas
│
├── services/                  # Servicios externos
│   ├── vertex_service.py      # Cliente para Vertex AI
│   └── storage_service.py     # Abstracción de almacenamiento
│
├── utils/                     # Utilidades
│   ├── chunking.py            # División de texto en chunks
│   ├── validators.py          # Validación de PDFs y queries
│   └── cache.py               # Sistema de caché de documentos
│
├── data/                      # Datos locales (desarrollo)
│   ├── pdfs/                  # PDFs de entrada
│   ├── outputs/               # Resultados procesados
│   └── temp/                  # Archivos temporales
│
└── deploy/                    # Configuración de deployment
    ├── cloudbuild.yaml        # Cloud Build pipeline
    └── service.yaml           # Cloud Run service config
```

## Instalación

### Requisitos Previos

- Python 3.11+
- Cuenta de Google Cloud Platform con billing habilitado
- Proyecto GCP con las siguientes APIs habilitadas:
  - Vertex AI API
  - Cloud Storage API
  - Cloud Run API (solo para producción)

### Instalación Local

#### 1. Clonar el repositorio

```bash
cd pry_pdf_resume
```

#### 2. Crear y activar entorno virtual

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Configurar GCP

##### a) Autenticación con GCP

```bash
gcloud auth application-default login
gcloud config set project TU_PROJECT_ID
```

##### b) Habilitar APIs necesarias

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
```

#### 5. Configurar variables de entorno

```bash
cp .env.development .env
```

Editar `.env` con tus credenciales:

```env
# Environment
ENVIRONMENT=development

# Google Cloud
GCP_PROJECT_ID=tu-proyecto-gcp
GCP_LOCATION=us-central1

# Vertex AI
VERTEX_AI_MODEL=gemini-2.5-pro
EMBEDDING_MODEL=text-embedding-004

# Storage
STORAGE_TYPE=local
LOCAL_PDF_PATH=./data/pdfs
LOCAL_OUTPUT_PATH=./data/outputs
LOCAL_TEMP_PATH=./data/temp

# Processing
MAX_PDF_PAGES=2000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONCURRENT_REQUESTS=5

# Application
PORT=8501
```

#### 6. Crear directorios necesarios

```bash
mkdir -p data/pdfs data/outputs data/temp
```

## Uso

### Desarrollo Local

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Uso de la Aplicación

#### 1. Cargar y Procesar Documento

1. Navega a la interfaz principal
2. Carga un archivo PDF (máximo 2000 páginas)
3. Selecciona el nivel de detalle del resumen:
   - **Ejecutivo**: Resumen conciso para ejecutivos
   - **Normal**: Balance entre detalle y brevedad
   - **Detallado**: Análisis exhaustivo con datos técnicos
4. Haz clic en "Procesar Documento"

El sistema realizará:
- Validación del PDF
- Extracción de texto de todas las páginas
- Detección automática de la estructura (TOC y secciones)
- Generación de resúmenes por sección
- Creación de embeddings para búsqueda

#### 2. Visualizar Resúmenes

Después del procesamiento, podrás ver:
- Resumen ejecutivo del documento completo
- Resúmenes organizados por capítulos y secciones
- Estructura jerárquica del documento
- Opción de descarga en formato JSON o TXT

#### 3. Hacer Preguntas (Q&A)

1. Ve a la pestaña "Preguntas y Respuestas"
2. Escribe tu pregunta sobre el contenido del documento
3. El sistema:
   - Buscará los fragmentos más relevantes
   - Generará una respuesta contextual
   - Proporcionará referencias a secciones específicas

## Deployment en Google Cloud

### Preparación

#### 1. Crear bucket de Cloud Storage

```bash
gsutil mb gs://TU_PROJECT_ID-pdfs
gsutil mb gs://TU_PROJECT_ID-outputs
```

#### 2. Configurar variables de entorno para producción

Editar `.env.production`:

```env
ENVIRONMENT=production
GCP_PROJECT_ID=tu-proyecto-gcp
GCP_LOCATION=us-central1
VERTEX_AI_MODEL=gemini-2.5-pro
EMBEDDING_MODEL=text-embedding-004
STORAGE_TYPE=cloud
GCS_BUCKET_NAME=tu-proyecto-gcp-pdfs
GCS_OUTPUT_BUCKET=tu-proyecto-gcp-outputs
```

### Deploy con Cloud Build

```bash
gcloud builds submit --config deploy/cloudbuild.yaml
```

### Deploy Manual con Docker y Cloud Run

#### 1. Build de la imagen

```bash
# Autenticar con Container Registry
gcloud auth configure-docker

# Build de la imagen
docker build -t gcr.io/TU_PROJECT_ID/pdf-rag-system:latest .

# Push de la imagen
docker push gcr.io/TU_PROJECT_ID/pdf-rag-system:latest
```

#### 2. Deploy en Cloud Run

```bash
gcloud run deploy pdf-rag-system \
  --image gcr.io/TU_PROJECT_ID/pdf-rag-system:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars GCP_PROJECT_ID=TU_PROJECT_ID \
  --set-env-vars GCP_LOCATION=us-central1 \
  --set-env-vars VERTEX_AI_MODEL=gemini-2.5-pro \
  --set-env-vars EMBEDDING_MODEL=text-embedding-004 \
  --set-env-vars STORAGE_TYPE=cloud \
  --set-env-vars GCS_BUCKET_NAME=TU_PROJECT_ID-pdfs \
  --set-env-vars GCS_OUTPUT_BUCKET=TU_PROJECT_ID-outputs \
  --allow-unauthenticated
```

## Configuración Avanzada

### Ajuste de Parámetros de Chunking

Editar en `config.py` o variables de entorno:

```python
CHUNK_SIZE=1000          # Tamaño de cada chunk en caracteres
CHUNK_OVERLAP=200        # Superposición entre chunks
```

Recomendaciones:
- PDFs técnicos: `CHUNK_SIZE=1500`, `CHUNK_OVERLAP=300`
- PDFs narrativos: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`

### Límites y Concurrencia

```python
MAX_PDF_PAGES=2000              # Máximo de páginas por PDF
MAX_CONCURRENT_REQUESTS=5        # Requests concurrentes a Vertex AI
```

### Niveles de Detalle de Resúmenes

Personalizar los prompts en `core/summary_generator.py`:

- **Ejecutivo**: Enfocado en insights clave y conclusiones
- **Normal**: Balance entre detalle y accesibilidad
- **Detallado**: Exhaustivo con datos técnicos y ejemplos

## Troubleshooting

### Error: "Cannot authenticate with GCP"

```bash
gcloud auth application-default login
gcloud config set project TU_PROJECT_ID
```

### Error: "Vertex AI API not enabled"

```bash
gcloud services enable aiplatform.googleapis.com
```

### Error: "Cloud Storage API not enabled"

```bash
gcloud services enable storage.googleapis.com
```

### PDF no procesa correctamente

**Síntomas**: El PDF no se procesa o genera errores

**Soluciones**:
1. Verificar que el PDF no esté protegido con contraseña
2. Asegurar que el PDF tenga texto extraíble (no escaneado como imagen)
3. Revisar logs en modo debug:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

### La detección de estructura falla

**Síntomas**: No se detecta la tabla de contenidos o las secciones

**Soluciones**:
1. El PDF debe tener una tabla de contenidos clara
2. Verificar que los títulos de capítulos sigan un patrón consistente
3. Si no hay TOC, el sistema usará análisis heurístico (menos preciso)

### Errores de cuota de Vertex AI

**Síntomas**: "Quota exceeded" o "Rate limit"

**Soluciones**:
1. Reducir `MAX_CONCURRENT_REQUESTS` en config
2. Solicitar aumento de cuota en GCP Console
3. Implementar retry logic con backoff exponencial

## Desarrollo

### Estructura de Logs

Todos los módulos usan logging de Python:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Mensaje informativo")
logger.warning("Mensaje de advertencia")
logger.error("Mensaje de error", exc_info=True)
```

### Ejecutar en modo debug

```bash
streamlit run app.py --logger.level=debug
```

### Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar con coverage
pytest --cov=. tests/
```

### Formateo de código

```bash
# Formatear con black
black .

# Verificar estilo con flake8
flake8 .

# Type checking con mypy
mypy .
```
## Licencia

Este proyecto está bajo licencia MIT. Ver archivo `LICENSE` para más detalles.

**Implementación:** Código generado con Claude AI (Anthropic) bajo supervisión y especificaciones del autor

**Nota sobre el uso de IA:** Este proyecto fue desarrollado mediante pair programming con IA. El diseño, la arquitectura y las decisiones técnicas son del autor humano. La implementación del código fue asistida por Claude AI siguiendo las especificaciones proporcionadas.

## Agradecimientos

- [Anthropic](https://anthropic.com) por Claude AI, asistente en el desarrollo del código
- **Google Cloud Vertex AI** por proporcionar modelos de IA de última generación
- **Streamlit** por el framework de desarrollo de aplicaciones
- **PyPDF2 y pdfplumber** communities por las herramientas de procesamiento de PDFs
- Comunidad de código abierto por las bibliotecas y herramientas utilizadas
