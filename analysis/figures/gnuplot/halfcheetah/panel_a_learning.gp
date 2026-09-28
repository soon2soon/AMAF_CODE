# HalfCheetah Panel A: TD3 vs AMAF learning dynamics.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/halfcheetah/panel_a_learning.gp

load "analysis/figures/gnuplot/amaf_common.gp"

datafile = "analysis/figures/figure_data/halfcheetah_panel_a_learning.csv"
pdf_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_a_learning.pdf"
png_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_a_learning.png"

set datafile separator comma

set xrange [0:1.5]
set xtics 0.25
set format y "%.0f"

# Shared compact horizontal margins for both rows.
set lmargin at screen 0.155
set rmargin at screen 0.985

# The delta strip is 31.9% of the upper axes height (0.185 / 0.580).
# Rendering commands are repeated so PDF and PNG use identical screen geometry.
set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 4.50in,3.40in
set output pdf_output
set multiplot

set bmargin at screen 0.400
set tmargin at screen 0.980
set yrange [-1000:14000]
set ytics 2000
set format x ""
unset xlabel
set ylabel "Evaluation return" offset 1.0,0
set key top left at graph 0.025,0.975
plot \
    datafile using ($1/1e6):3:4 \
        with filledcurves fillcolor rgb td3_color fillstyle transparent solid 0.16 noborder notitle, \
    datafile using ($1/1e6):6:7 \
        with filledcurves fillcolor rgb amaf_color fillstyle transparent solid 0.14 noborder notitle, \
    datafile using ($1/1e6):2 with lines linestyle 1 notitle, \
    datafile using ($1/1e6):5 with lines linestyle 2 notitle, \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $2 : 1/0) \
        with linespoints linestyle 11 title "TD3", \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $5 : 1/0) \
        with linespoints linestyle 12 title "AMAF"

set bmargin at screen 0.135
set tmargin at screen 0.320
set yrange [-2000:1600]
set ytics 1000
set format x "%g"
set xlabel "Environment steps (millions)" offset 0,0.25
set ylabel "AMAF - TD3" offset 1.0,0
unset key
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linestyle 20
set arrow 2 from first 1.0, graph 0 to first 1.0, graph 1 nohead back linestyle 21
plot \
    datafile using ($1/1e6):9:10 \
        with filledcurves fillcolor rgb amaf_color fillstyle transparent solid 0.13 noborder notitle, \
    datafile using ($1/1e6):8 with lines linestyle 2 notitle, \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $8 : 1/0) \
        with linespoints linestyle 12 notitle

unset arrow 1
unset arrow 2
unset multiplot
unset output

set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1350,1020
set output png_output
set multiplot

set bmargin at screen 0.400
set tmargin at screen 0.980
set yrange [-1000:14000]
set ytics 2000
set format x ""
unset xlabel
set ylabel "Evaluation return" offset 1.0,0
set key top left at graph 0.025,0.975
plot \
    datafile using ($1/1e6):3:4 \
        with filledcurves fillcolor rgb td3_color fillstyle transparent solid 0.16 noborder notitle, \
    datafile using ($1/1e6):6:7 \
        with filledcurves fillcolor rgb amaf_color fillstyle transparent solid 0.14 noborder notitle, \
    datafile using ($1/1e6):2 with lines linestyle 1 notitle, \
    datafile using ($1/1e6):5 with lines linestyle 2 notitle, \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $2 : 1/0) \
        with linespoints linestyle 11 title "TD3", \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $5 : 1/0) \
        with linespoints linestyle 12 title "AMAF"

set bmargin at screen 0.135
set tmargin at screen 0.320
set yrange [-2000:1600]
set ytics 1000
set format x "%g"
set xlabel "Environment steps (millions)" offset 0,0.25
set ylabel "AMAF - TD3" offset 1.0,0
unset key
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linestyle 20
set arrow 2 from first 1.0, graph 0 to first 1.0, graph 1 nohead back linestyle 21
plot \
    datafile using ($1/1e6):9:10 \
        with filledcurves fillcolor rgb amaf_color fillstyle transparent solid 0.13 noborder notitle, \
    datafile using ($1/1e6):8 with lines linestyle 2 notitle, \
    datafile using ($1/1e6):((int($1/250000.0) == $1/250000.0) ? $8 : 1/0) \
        with linespoints linestyle 12 notitle

unset arrow 1
unset arrow 2
unset multiplot
unset output
