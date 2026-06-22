$ErrorActionPreference = "Stop"

$papers = @(
  @{ Id = "2604.26779"; File = "2604.26779_RL_Post_Training_Rollouts_System_Integrated_Speculative_Decoding.pdf" },
  @{ Id = "2605.04263"; File = "2605.04263_PARSE_Parallel_Prefix_Verification_for_Speculative_Generation.pdf" },
  @{ Id = "2605.20104"; File = "2605.20104_Graft_Draft_Less_Retrieve_More.pdf" },
  @{ Id = "2605.09992"; File = "2605.09992_Attention_Drift_Autoregressive_Speculative_Decoding.pdf" },
  @{ Id = "2605.02888"; File = "2605.02888_SpecKV_Adaptive_Speculative_Decoding.pdf" }
)

$targetDir = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($paper in $papers) {
  $url = "https://arxiv.org/pdf/$($paper.Id)"
  $out = Join-Path $targetDir $paper.File
  Write-Host "Downloading $url -> $out"
  Invoke-WebRequest -Uri $url -OutFile $out
}

Write-Host "Done."
