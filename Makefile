.PHONY: test dry-run figures discussion-data verify-data

test:
	pytest -q

dry-run:
	amaf-run --experiment experiments/revision/lunarlander_fixed500k.yaml --dry-run

figures:
	bash scripts/render_paper_figures.sh

discussion-data:
	python -m analysis.figures.export_discussion_figures

verify-data:
	sha256sum -c DATA_SHA256SUMS
