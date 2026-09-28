# HalfCheetah Panel B: seed-level relative-gap transition.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/halfcheetah/panel_b_transition.gp

load "analysis/figures/gnuplot/amaf_common.gp"

datafile = "analysis/figures/figure_data/halfcheetah_panel_b_transition.csv"
pdf_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_b_transition.pdf"
png_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_b_transition.png"

set datafile separator comma
set xrange [-4200:2200]
set yrange [-4200:2200]
set xtics 1000
set ytics 1000
set format x "%.0f"
set format y "%.0f"
set xlabel "AMAF - TD3 at 1.0M"
set ylabel "AMAF - TD3 at 1.5M" offset 0.6,0
set key top left at graph 0.025,0.975
set size ratio -1

set lmargin at screen 0.205
set rmargin at screen 0.975
set bmargin at screen 0.150
set tmargin at screen 0.980

set arrow 1 from first 0, graph 0 to first 0, graph 1 nohead back linestyle 22
set arrow 2 from graph 0, first 0 to graph 1, first 0 nohead back linestyle 22

set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 3.50in,3.25in
set output pdf_output
plot \
    x with lines linestyle 21 title "y = x", \
    datafile using (strcol(2) eq "original5" && $6 == 1 ? $3 : 1/0):4 \
        with points linestyle 31 title "Original 5", \
    datafile using (strcol(2) eq "replication5" && $6 == 1 ? $3 : 1/0):4 \
        with points linestyle 32 title "Replication 5", \
    datafile using (strcol(2) eq "original5" && $6 == 0 ? $3 : 1/0):4 \
        with points linestyle 33 notitle, \
    datafile using (strcol(2) eq "replication5" && $6 == 0 ? $3 : 1/0):4 \
        with points linestyle 34 notitle
unset output

set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1050,975
set output png_output
replot
unset output
