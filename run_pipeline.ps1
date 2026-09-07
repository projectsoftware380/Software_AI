# =========================================================================
# Build + push de imagen y lanzamiento de pipeline en Vertex AI
# =========================================================================

function Remove-ExistingImages {
    param(
        [string]$ProjectID,
        [string]$Region,
        [string]$RepoName,
        [string]$ImageName
    )

    $RepoUrl = "${Region}-docker.pkg.dev/${ProjectID}/${RepoName}"
    Write-Host "Buscando imágenes existentes en '$RepoUrl'..."

    $images = gcloud artifacts docker images list $RepoUrl --filter="package~${ImageName}" --format="get(image)" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "No fue posible listar imágenes existentes; se continuará sin limpieza previa."
        return
    }

    if ($images) {
        $imageList = $images -split "`n" | ForEach-Object { $_.Trim() }
        foreach ($imageUri in $imageList) {
            if ($imageUri) {
                Write-Host "Eliminando imagen: $imageUri"
                gcloud artifacts docker images delete $imageUri --quiet
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "No se pudo eliminar la imagen: $imageUri"
                }
            }
        }
    }
}

# Configuración por entorno. No se incluyen IDs reales en el repositorio.
$ProjectID = $env:GCP_PROJECT_ID
$Region = if ($env:GCP_REGION) { $env:GCP_REGION } else { "europe-west1" }
$RepoName = if ($env:ARTIFACT_REPOSITORY) { $env:ARTIFACT_REPOSITORY } else { "mlops-images" }
$ImageName = if ($env:PIPELINE_IMAGE_NAME) { $env:PIPELINE_IMAGE_NAME } else { "software-ai-pipeline" }

if (-not $ProjectID) {
    Write-Error "Falta GCP_PROJECT_ID. Defínelo antes de ejecutar este script."
    exit 1
}

Remove-ExistingImages -ProjectID $ProjectID -Region $Region -RepoName $RepoName -ImageName $ImageName

$VersionTag = Get-Date -Format "yyyyMMdd-HHmmss"
$ImageUri = "${Region}-docker.pkg.dev/${ProjectID}/${RepoName}/${ImageName}:${VersionTag}"

Write-Host "Construyendo imagen: $ImageUri"
docker build --no-cache -t $ImageUri .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falló la construcción de la imagen Docker."
    exit 1
}

Write-Host "Subiendo imagen a Artifact Registry..."
docker push $ImageUri
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falló la subida de la imagen."
    exit 1
}

# Usa el intérprete del entorno virtual si existe; de lo contrario usa Python
# disponible en PATH.
$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} elseif (Test-Path ".\venv\Scripts\python.exe") {
    ".\venv\Scripts\python.exe"
} else {
    "python"
}

Write-Host "Lanzando pipeline con $PythonExe ..."
& $PythonExe -m src.pipeline.main --common-image-uri $ImageUri
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falló el lanzamiento de la pipeline."
    exit 1
}

Write-Host "Pipeline lanzada con la imagen $ImageUri"
