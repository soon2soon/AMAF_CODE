# Lunar F4: raw seed evidence with method means and bootstrap intervals.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/lunar/f4_method_summary.gp

load "analysis/figures/gnuplot/amaf_common.gp"

summary_file = "analysis/figures/figure_data/lunar_f4_method_summary.csv"
points_file = "analysis/figures/figure_data/lunar_f4_method_points.csv"
pdf_output = "paper_figures/gnuplot/lunar/lunar_f4_method_summary.pdf"
png_output = "paper_figures/gnuplot/lunar/lunar_f4_method_summary.png"

set datafile separator comma
unset key
unset colorbox
unset grid
set grid xtics ytics back linestyle 90
set yrange [0.55:5.45]
set ytics ("Protected" 1, "Naive" 2, "Uniform" 3, "Dueling" 4, "DQN" 5)
set mytics 1
set bars small 1.2

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,3.20in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,960
        set output png_output
    }

    set multiplot

    set lmargin at screen 0.105
    set rmargin at screen 0.490
    set bmargin at screen 0.180
    set tmargin at screen 0.915
    set title "(a) Late-stage performance" font "Liberation Sans Bold,11.5" offset 0,-0.2
    set xlabel "Last-5 return" offset 0,0.25
    set xrange [-140:320]
    set xtics 100
    set mxtics 2
    plot \
        points_file using (strcol(1) eq "DQN" ? $5 : 1/0):4 with points linestyle 64 pointsize 0.58 notitle, \
        points_file using (strcol(1) eq "Dueling" ? $5 : 1/0):4 with points linestyle 65 pointsize 0.56 notitle, \
        points_file using (strcol(1) eq "Uniform" ? $5 : 1/0):4 with points linestyle 61 pointsize 0.62 notitle, \
        points_file using (strcol(1) eq "Naive" ? $5 : 1/0):4 with points linestyle 62 pointsize 0.68 notitle, \
        points_file using (strcol(1) eq "Protected" ? $5 : 1/0):4 with points linestyle 63 pointsize 0.64 notitle, \
        summary_file using (strcol(1) eq "DQN" && strcol(3) eq "last5" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_dqn_color linewidth 1.25 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Dueling" && strcol(3) eq "last5" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_dueling_color linewidth 1.20 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Uniform" && strcol(3) eq "last5" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_uniform_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Naive" && strcol(3) eq "last5" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_naive_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Protected" && strcol(3) eq "last5" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_protected_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "DQN" && strcol(3) eq "last5" ? $5 : 1/0):2 with points linestyle 64 pointsize 1.28 linewidth 1.6 notitle, \
        summary_file using (strcol(1) eq "Dueling" && strcol(3) eq "last5" ? $5 : 1/0):2 with points linestyle 65 pointsize 1.28 linewidth 1.6 notitle, \
        summary_file using (strcol(1) eq "Uniform" && strcol(3) eq "last5" ? $5 : 1/0):2 with points linestyle 61 pointsize 1.34 linewidth 1.7 notitle, \
        summary_file using (strcol(1) eq "Naive" && strcol(3) eq "last5" ? $5 : 1/0):2 with points linestyle 62 pointsize 1.38 linewidth 1.8 notitle, \
        summary_file using (strcol(1) eq "Protected" && strcol(3) eq "last5" ? $5 : 1/0):2 with points linestyle 63 pointsize 1.34 linewidth 1.7 notitle

    set lmargin at screen 0.605
    set rmargin at screen 0.985
    set bmargin at screen 0.180
    set tmargin at screen 0.915
    set title "(b) Retention" font "Liberation Sans Bold,11.5" offset 0,-0.2
    set xlabel "Retention (%)" offset 0,0.25
    set xrange [-55:105]
    set xtics 25
    set mxtics 1
    plot \
        points_file using (strcol(1) eq "DQN" ? $6 : 1/0):4 with points linestyle 64 pointsize 0.58 notitle, \
        points_file using (strcol(1) eq "Dueling" ? $6 : 1/0):4 with points linestyle 65 pointsize 0.56 notitle, \
        points_file using (strcol(1) eq "Uniform" ? $6 : 1/0):4 with points linestyle 61 pointsize 0.62 notitle, \
        points_file using (strcol(1) eq "Naive" ? $6 : 1/0):4 with points linestyle 62 pointsize 0.68 notitle, \
        points_file using (strcol(1) eq "Protected" ? $6 : 1/0):4 with points linestyle 63 pointsize 0.64 notitle, \
        summary_file using (strcol(1) eq "DQN" && strcol(3) eq "retention_pct" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_dqn_color linewidth 1.25 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Dueling" && strcol(3) eq "retention_pct" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_dueling_color linewidth 1.20 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Uniform" && strcol(3) eq "retention_pct" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_uniform_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Naive" && strcol(3) eq "retention_pct" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_naive_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "Protected" && strcol(3) eq "retention_pct" ? $5 : 1/0):2:7:8 with xerrorbars linecolor rgb lunar_protected_color linewidth 1.55 pointtype -1 notitle, \
        summary_file using (strcol(1) eq "DQN" && strcol(3) eq "retention_pct" ? $5 : 1/0):2 with points linestyle 64 pointsize 1.28 linewidth 1.6 notitle, \
        summary_file using (strcol(1) eq "Dueling" && strcol(3) eq "retention_pct" ? $5 : 1/0):2 with points linestyle 65 pointsize 1.28 linewidth 1.6 notitle, \
        summary_file using (strcol(1) eq "Uniform" && strcol(3) eq "retention_pct" ? $5 : 1/0):2 with points linestyle 61 pointsize 1.34 linewidth 1.7 notitle, \
        summary_file using (strcol(1) eq "Naive" && strcol(3) eq "retention_pct" ? $5 : 1/0):2 with points linestyle 62 pointsize 1.38 linewidth 1.8 notitle, \
        summary_file using (strcol(1) eq "Protected" && strcol(3) eq "retention_pct" ? $5 : 1/0):2 with points linestyle 63 pointsize 1.34 linewidth 1.7 notitle

    unset multiplot
    unset output
}
