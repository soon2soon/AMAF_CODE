# Lunar F1: focal adaptive methods with DQN and Dueling context.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/lunar/f1_learning.gp

load "analysis/figures/gnuplot/amaf_common.gp"

datafile = "analysis/figures/figure_data/lunar_f1_learning.csv"
if (!exists("pdf_output")) pdf_output = "paper_figures/gnuplot/lunar/lunar_f1_learning.pdf"
if (!exists("png_output")) png_output = "paper_figures/gnuplot/lunar/lunar_f1_learning.png"
# Default hierarchy: focal-method CIs remain visible while the two reference
# baselines retain complete mean trajectories. Their CI columns stay exported;
# this is uncertainty emphasis for the focal comparison, not claim selection.
if (!exists("show_reference_ci")) show_reference_ci = 0

# Ten markers per 500k curve: visible at manuscript width without crowding.
marker_step = 50000
marker_y(step, value) = (int(step / marker_step) == step / marker_step ? value : 1/0)

set datafile separator comma
set xrange [0:0.5]
set yrange [-700:320]
set xtics 0.1
set ytics 200
set format x "%g"
set format y "%.0f"
set xlabel "Environment steps (millions)" offset 0,0.25
set ylabel "Evaluation return" offset 1.0,0
set key bottom right at graph 0.975,0.035 maxrows 3

set lmargin at screen 0.155
set rmargin at screen 0.985
set bmargin at screen 0.140
set tmargin at screen 0.980

set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 3.70in,2.75in
set output pdf_output
plot \
    datafile using ($1/1e6):(show_reference_ci ? $12 : 1/0):(show_reference_ci ? $13 : 1/0) \
        with filledcurves fillcolor rgb lunar_dqn_color fillstyle transparent solid 0.045 noborder notitle, \
    datafile using ($1/1e6):(show_reference_ci ? $15 : 1/0):(show_reference_ci ? $16 : 1/0) \
        with filledcurves fillcolor rgb lunar_dueling_color fillstyle transparent solid 0.045 noborder notitle, \
    datafile using ($1/1e6):3:4 \
        with filledcurves fillcolor rgb lunar_uniform_color fillstyle transparent solid 0.11 noborder notitle, \
    datafile using ($1/1e6):6:7 \
        with filledcurves fillcolor rgb lunar_naive_color fillstyle transparent solid 0.08 noborder notitle, \
    datafile using ($1/1e6):9:10 \
        with filledcurves fillcolor rgb lunar_protected_color fillstyle transparent solid 0.11 noborder notitle, \
    datafile using ($1/1e6):11 with lines linestyle 64 notitle, \
    datafile using ($1/1e6):14 with lines linestyle 65 notitle, \
    datafile using ($1/1e6):2 with lines linestyle 61 notitle, \
    datafile using ($1/1e6):5 with lines linestyle 62 notitle, \
    datafile using ($1/1e6):8 with lines linestyle 63 notitle, \
    datafile using ($1/1e6):(marker_y($1, $2)) \
        with linespoints linestyle 61 title "Uniform", \
    datafile using ($1/1e6):(marker_y($1, $5)) \
        with linespoints linestyle 62 title "Naive", \
    datafile using ($1/1e6):(marker_y($1, $8)) \
        with linespoints linestyle 63 title "Protected", \
    datafile using ($1/1e6):(marker_y($1, $11)) \
        with linespoints linestyle 64 title "DQN", \
    datafile using ($1/1e6):(marker_y($1, $14)) \
        with linespoints linestyle 65 title "Dueling"
unset output

set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1110,825
set output png_output
replot
unset output
