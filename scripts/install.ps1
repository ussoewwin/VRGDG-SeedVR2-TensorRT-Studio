param(
    [switch]$SkipModels,
    [switch]$SkipTensorRT,
    [switch]$Repair
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$StudioRoot = Split-Path -Parent $PSScriptRoot
$Outputs = Join-Path $StudioRoot 'outputs'
$Log = Join-Path $Outputs 'install.log'
$Venv = Join-Path $StudioRoot '.venv'
$StudioPython = Join-Path $Venv 'Scripts\python.exe'
$Marker = Join-Path $StudioRoot '.seedvr-studio-installed'
Push-Location -LiteralPath $StudioRoot

New-Item -ItemType Directory -Force $Outputs | Out-Null
try { Start-Transcript -Path $Log -Append | Out-Null } catch {}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host ('== ' + $Message + ' ==') -ForegroundColor Cyan
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = $machinePath + ';' + $userPath

    $candidatePaths = @(
        'C:\Program Files\ffmpeg\bin',
        'C:\Program Files\ffmpeg',
        'C:\Program Files (x86)\ffmpeg\bin',
        'C:\ffmpeg\bin',
        'D:\ffmpeg\bin',
        (Join-Path $StudioRoot 'bin\ffmpeg\bin'),
        (Join-Path $StudioRoot 'bin')
    )
    foreach ($p in $candidatePaths) {
        if ((Test-Path (Join-Path $p 'ffmpeg.exe')) -and (Test-Path (Join-Path $p 'ffprobe.exe'))) {
            $env:Path = $p + ';' + $env:Path
            break
        }
    }
}

function Find-Python314 {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'SilentlyContinue'
            $path = & $py.Source -3.14 -c "import sys; print(sys.executable)" 2>$null
            $pyExitCode = $LASTEXITCODE
        }
        catch {
            $path = $null
            $pyExitCode = 1
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($pyExitCode -eq 0 -and $path) { return ($path | Select-Object -Last 1).Trim() }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'SilentlyContinue'
            $version = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $pythonExitCode = $LASTEXITCODE
        }
        catch {
            $version = $null
            $pythonExitCode = 1
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($pythonExitCode -eq 0 -and $version -and [version]$version -eq [version]'3.14') { return $python.Source }
    }
    $locations = @(
        (Join-Path $env:LocalAppData 'Programs\Python\Python314\python.exe'),
        (Join-Path $env:ProgramFiles 'Python314\python.exe'),
        'C:\Python314\python.exe',
        'D:\Python314\python.exe'
    )
    return $locations | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Ensure-FFmpeg {
    if ((Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -and (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
        return
    }

    $candidatePaths = @(
        'C:\Program Files\ffmpeg\bin',
        'C:\Program Files\ffmpeg',
        'C:\Program Files (x86)\ffmpeg\bin',
        'C:\ffmpeg\bin',
        'D:\ffmpeg\bin',
        (Join-Path $StudioRoot 'bin\ffmpeg\bin'),
        (Join-Path $StudioRoot 'bin')
    )

    foreach ($p in $candidatePaths) {
        if ((Test-Path (Join-Path $p 'ffmpeg.exe')) -and (Test-Path (Join-Path $p 'ffprobe.exe'))) {
            $env:Path = $p + ';' + $env:Path
            Write-Host "Found FFmpeg at: $p" -ForegroundColor Green
            return
        }
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Step 'Installing FFmpeg via winget'
        & $winget.Source install --id 'Gyan.FFmpeg' --exact --accept-package-agreements --accept-source-agreements --silent
        Refresh-Path
        if ((Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -and (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
            return
        }
    }

    Write-Step 'Downloading FFmpeg portable build'
    $ffmpegDir = Join-Path $StudioRoot 'bin\ffmpeg'
    $ffmpegZip = Join-Path $StudioRoot 'bin\ffmpeg.zip'
    New-Item -ItemType Directory -Force (Join-Path $StudioRoot 'bin') | Out-Null
    
    $downloadUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $ffmpegZip -UseBasicParsing
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force
    Remove-Item -Force $ffmpegZip

    $extractedBin = Get-ChildItem -Path $ffmpegDir -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
    if ($extractedBin) {
        $binDir = $extractedBin.DirectoryName
        $env:Path = $binDir + ';' + $env:Path
        Write-Host "FFmpeg installed to: $binDir" -ForegroundColor Green
    } else {
        throw "Could not locate ffmpeg.exe after extracting to $ffmpegDir"
    }
}

function Ensure-Python314 {
    $basePy = Find-Python314
    if ($basePy) { return $basePy }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Step 'Installing Python 3.14 via winget'
        & $winget.Source install --id 'Python.Python.3.14' --exact --accept-package-agreements --accept-source-agreements --silent
        Refresh-Path
        $basePy = Find-Python314
        if ($basePy) { return $basePy }
    }

    $localInstallers = @(
        'D:\Python_(64bit)_v3.14.7.exe',
        'D:\Python_(64bit)_v3.14.6.exe'
    )
    foreach ($inst in $localInstallers) {
        if (Test-Path -LiteralPath $inst) {
            Write-Step "Installing Python 3.14 from $inst"
            Start-Process -FilePath $inst -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_pip=1' -Wait
            Refresh-Path
            $basePy = Find-Python314
            if ($basePy) { return $basePy }
        }
    }

    Write-Step 'Downloading Python 3.14 installer'
    $pyInstaller = Join-Path $Outputs 'python-3.14.0-amd64.exe'
    $downloadUrl = 'https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $pyInstaller -UseBasicParsing
    
    Write-Step 'Installing Python 3.14'
    Start-Process -FilePath $pyInstaller -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_pip=1' -Wait
    Remove-Item -Force $pyInstaller -ErrorAction SilentlyContinue
    Refresh-Path

    $basePy = Find-Python314
    if (-not $basePy) {
        throw 'Python 3.14 was installed but could not be located. Restart Windows and run the installer again.'
    }
    return $basePy
}

function Install-WingetPackage([string]$Id, [string]$Name) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$Name is missing and Windows Package Manager (winget) is unavailable. Install $Name manually, then run this installer again."
    }
    Write-Step "Installing $Name"
    & $winget.Source install --id $Id --exact --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Name (exit $LASTEXITCODE)." }
    Refresh-Path
}

function Invoke-Pip([string[]]$Arguments) {
    & $StudioPython -m pip --isolated @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Arguments -join ' ')" }
}

try {
    Write-Host 'SeedVR Studio complete Windows installer' -ForegroundColor Green
    Write-Host 'Installs the UI, CUDA PyTorch, SeedVR2, SageAttention 2, TensorRT RTX, models, FFmpeg, and GPU-specific engines.'
    Write-Host 'Interrupted downloads and engine builds can be resumed by running this installer again.'

    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        throw 'No NVIDIA driver was detected. Install a current NVIDIA driver, restart Windows, and run this installer again.'
    }

    Ensure-FFmpeg

    $BasePython = Ensure-Python314

    Write-Step 'Creating the private Python environment'
    if ($Repair -and (Test-Path $Venv)) {
        Write-Warning 'Repair mode reuses the existing environment and reinstalls all packages.'
    }
    if (-not (Test-Path $StudioPython)) {
        & $BasePython -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    }
    Invoke-Pip @('install', '--upgrade', 'pip', 'setuptools', 'wheel', '--index-url', 'https://pypi.org/simple')

    Write-Step 'Installing CUDA 13.2 PyTorch build (torch 2.13.0+cu132)'
    $torchArguments = @(
        '-m', 'pip', '--isolated', 'install',
        'torch==2.13.0+cu132',
        'torchvision',
        '--index-url', 'https://download.pytorch.org/whl/cu132'
    )
    & $StudioPython @torchArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Installing torch and torchvision from cu132 repository...'
        Invoke-Pip @('install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu132')
    }

    Write-Step 'Installing SeedVR Studio and SeedVR2 dependencies'
    Invoke-Pip @('install', '--editable', $StudioRoot, '--index-url', 'https://pypi.org/simple')
    Invoke-Pip @('install', '--requirement', (Join-Path $StudioRoot 'vendor\seedvr2\requirements.txt'), '--index-url', 'https://pypi.org/simple')

    function Install-CachedWheel([string]$ModuleName, [string]$PackageName, [string]$Url) {
        $check = & $StudioPython -c "import $ModuleName" 2>$null
        if ($LASTEXITCODE -eq 0 -and -not $Repair) {
            Write-Host "$PackageName is already installed." -ForegroundColor Green
            return
        }

        $wheelsDir = Join-Path $StudioRoot 'wheels'
        New-Item -ItemType Directory -Force $wheelsDir | Out-Null
        $filename = [System.IO.Path]::GetFileName($Url.Split('?')[0])
        $filename = [System.Uri]::UnescapeDataString($filename)
        $cachedPath = Join-Path $wheelsDir $filename

        if (-not (Test-Path -LiteralPath $cachedPath)) {
            Write-Step "Downloading $PackageName wheel"
            Invoke-WebRequest -Uri $Url -OutFile $cachedPath -UseBasicParsing
        }

        Write-Step "Installing $PackageName"
        Invoke-Pip @('install', $cachedPath)
    }

    Write-Step 'Installing FlashAttention 2, SageAttention 2, and TensorRT RTX'
    Install-CachedWheel 'flash_attn' 'FlashAttention 2' 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.4%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl'
    Install-CachedWheel 'sageattention' 'SageAttention 2' 'https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl'
    Invoke-Pip @('install', '--requirement', (Join-Path $StudioRoot 'requirements-windows-cu130.txt'), '--index-url', 'https://pypi.org/simple')

    Write-Step 'Checking CUDA and attention support'
    & $StudioPython -c "import sys, torch; sys.path.insert(0, 'vendor/seedvr2'); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE, FLASH_ATTN_2_AVAILABLE; print('SageAttention 2:', 'ready' if SAGE_ATTN_2_AVAILABLE else 'not available'); print('FlashAttention 2:', 'ready' if FLASH_ATTN_2_AVAILABLE else 'not available'); assert SAGE_ATTN_2_AVAILABLE or FLASH_ATTN_2_AVAILABLE, 'Neither SageAttention 2 nor FlashAttention 2 loaded'"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA or attention kernel verification failed. See docs\SAGEATTENTION.md.' }

    if (-not $SkipModels) {
        Write-Step 'Downloading the default SeedVR2 3B FP8 model and VAE'
        & $StudioPython (Join-Path $PSScriptRoot 'download_models.py')
        if ($LASTEXITCODE -ne 0) { throw 'The model download failed. Run this installer again to resume it.' }
    } else {
        Write-Warning 'Model download was skipped. Models will download when first selected.'
    }

    if (-not $SkipTensorRT) {
        Write-Step 'Building GPU-specific TensorRT VAE engines'
        Write-Host 'This is a one-time operation and may take a while. Do not copy these engines to a different GPU.'
        & $StudioPython (Join-Path $PSScriptRoot 'prepare_tensorrt.py')
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT engine preparation failed. Run this installer again to resume it.' }
    } else {
        Write-Warning 'TensorRT engine preparation was skipped. Only the Legacy engine will be usable.'
    }

    if (-not $SkipModels -and -not $SkipTensorRT) {
        Write-Step 'Final readiness check'
        Refresh-Path
        & $StudioPython (Join-Path $PSScriptRoot 'verify_install.py')
        if ($LASTEXITCODE -ne 0) { throw 'The final installation check failed.' }
        Set-Content -LiteralPath $Marker -Value (Get-Date -Format o) -Encoding ascii
    }

    Write-Host ''
    Write-Host 'SeedVR Studio is installed and ready.' -ForegroundColor Green
    Write-Host 'Double-click Launch SeedVR Studio Pro.bat to start.'
}
catch {
    Write-Host ''
    Write-Host ('INSTALLATION FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Detailed log: ' + $Log) -ForegroundColor Yellow
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Pop-Location } catch {}
}
