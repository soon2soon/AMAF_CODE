#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v gnuplot >/dev/null 2>&1; then
  echo "gnuplot is required (Debian/Ubuntu: apt-get install gnuplot-nox fonts-liberation)." >&2
  exit 1
fi

family="${1:-all}"
case "$family" in
  all|lunar|crop|halfcheetah|discussion) ;;
  *)
    echo "usage: $0 [all|lunar|crop|halfcheetah|discussion]" >&2
    exit 2
    ;;
esac

render_family() {
  local name="$1"
  mkdir -p "paper_figures/gnuplot/$name"
  while IFS= read -r script; do
    echo "gnuplot $script"
    gnuplot "$script"
  done < <(find "analysis/figures/gnuplot/$name" -maxdepth 1 -type f -name '*.gp' | sort)
}

if [[ "$family" == "all" ]]; then
  for name in lunar crop halfcheetah discussion; do
    render_family "$name"
  done
else
  render_family "$family"
fi

pdf_count="$(find paper_figures/gnuplot -type f -name '*.pdf' | wc -l)"
png_count="$(find paper_figures/gnuplot -type f -name '*.png' | wc -l)"
echo "Rendered figures: ${pdf_count} PDF, ${png_count} PNG"
