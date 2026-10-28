# Single entry point.
#   run.ps1 [reconstruct] [folder] [prompt]   transform the music in a folder (default)
#   run.ps1 setup                             create ./env and install the main dependencies
#   run.ps1 acestep-setup                     install ACE-Step 1.5 (code in ./ace-step, heavy files in MODELS_DIR)
#   run.ps1 acestep-cover [folder] [--strengths 0.5,0.7,1.0]
#   run.ps1 setup-all [--check] [--models a,b]   everything a new machine needs (+ health check)
#   run.ps1 ui [--port 7860] [--no-browser]   tuning UI: sliders + instant listening
#   run.ps1 check-tracks <folder> [--demux]   check a pack of reference tracks (+ split them into stems)
#   run.ps1 i18n update|check|new <lang>      UI translations (locales\<lang>.json), Qt style
#   run.ps1 native-bank <song-root>/reference [--redo]   slice reference stems into a bank of real samples (route B)
#   run.ps1 native-arrange <song-root> [--tracks a,b] [--start S --length L] [--seed N]   replay the song with them
#   run.ps1 pack-data --dest <folder> [--song <folder>] [--tracks <folder>] [--weights]   data for another machine
$ErrorActionPreference = 'Stop'

$commands = @{
    'reconstruct'   = 'reconstruct.ps1'
    'setup'         = 'setup_main.ps1'
    'acestep-setup' = 'setup_acestep.ps1'
    'acestep-cover' = 'acestep_cover.ps1'
    'ui'            = 'ui.ps1'
    'setup-all'     = 'setup_all.ps1'
    'check-tracks'  = 'check_tracks.ps1'
    'pack-data'     = 'pack_data.ps1'
    'native-bank'   = 'native_bank.ps1'
    'native-arrange' = 'native_arrange.ps1'
    'i18n'          = 'i18n.ps1'
}

$rest = $args
$name = 'reconstruct'
if ($args.Count -gt 0 -and $commands.ContainsKey([string]$args[0])) {
    $name = [string]$args[0]
    $rest = @(if ($args.Count -gt 1) { $args[1..($args.Count - 1)] })   # @() keeps a single argument an array
}
# otherwise a folder was dragged onto run.bat: treat all arguments as `reconstruct` arguments

& (Join-Path $PSScriptRoot "scripts\$($commands[$name])") @rest
