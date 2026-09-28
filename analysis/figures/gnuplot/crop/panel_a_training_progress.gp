# Crop Panel A: full and late-stage 100-episode-block learning curves.
# Python owns block aggregation and bootstrap statistics; gnuplot only renders.

load "analysis/figures/gnuplot/amaf_common.gp"

main_file = "analysis/figures/figure_data/crop_panel_a_main_curve.csv"
zoom_file = "analysis/figures/figure_data/crop_panel_a_zoom_curve.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_a_training_progress.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_a_training_progress.png"

set datafile separator comma
unset colorbox
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91
set xlabel font axis_label_font
set ylabel font axis_label_font

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,5.20in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1560
        set output png_output
    }

    set multiplot

    set key at graph 0.985,0.035 right bottom horizontal maxcols 3 opaque box linewidth 0.6 samplen 1.4 spacing 0.8 font "Liberation Sans,10"
    set lmargin at screen 0.125
    set rmargin at screen 0.985
    set bmargin at screen 0.370
    set tmargin at screen 0.925
    set title "Full training trajectory (Episodes 1–3000)" font "Liberation Sans Bold,11.5" offset 0,-0.2
    set xrange [0:3000]
    set yrange [-8300:1350]
    set xtics 500
    set mxtics 2
    set ytics 1000
    set mytics 2
    set xlabel ""
    set ylabel "Mean train return (100-episode block)" offset 1.0,0
    set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#424242" linewidth 1.00
    set arrow 2 from first 2000, graph 0 to first 2000, graph 1 nohead back linecolor rgb "#8A8A8A" linewidth 0.85 dashtype 2
    plot \
        main_file using (strcol(1) eq "Uniform" ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_uniform_color fillstyle transparent solid 0.10 noborder notitle, \
        main_file using (strcol(1) eq "Anchored" ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_anchored_color fillstyle transparent solid 0.10 noborder notitle, \
        main_file using (strcol(1) eq "DQN" ? $2 : 1/0):4 with linespoints linestyle 71 pointinterval 2 pointsize 0.60 title "DQN", \
        main_file using (strcol(1) eq "Dueling" ? $2 : 1/0):4 with linespoints linestyle 72 pointinterval 2 pointsize 0.64 title "Dueling", \
        main_file using (strcol(1) eq "Naive" ? $2 : 1/0):4 with linespoints linestyle 74 pointinterval 2 pointsize 0.70 title "Naive", \
        main_file using (strcol(1) eq "Uniform" ? $2 : 1/0):4 with linespoints linestyle 73 pointinterval 2 pointsize 0.72 title "Uniform", \
        main_file using (strcol(1) eq "Anchored" ? $2 : 1/0):4 with linespoints linestyle 75 pointinterval 2 pointsize 0.78 title "Anchored"

    unset key
    unset arrow 1
    unset arrow 2
    set lmargin at screen 0.125
    set rmargin at screen 0.985
    set bmargin at screen 0.105
    set tmargin at screen 0.280
    set title "Late-stage 100-episode block means (Episodes 2001–3000)" font "Liberation Sans Bold,11.5" offset 0,-0.15
    set xrange [2050:3050]
    set yrange [580:1160]
    set xtics ("2100" 2100, "2200" 2200, "2300" 2300, "2400" 2400, "2500" 2500, "2600" 2600, "2700" 2700, "2800" 2800, "2900" 2900, "3000" 3000)
    set mxtics 1
    set ytics 100
    set mytics 2
    set xlabel "Episode block end" offset 0,0.25
    set ylabel "Mean train return" offset 1.0,0
    set boxwidth 14 absolute
    set style fill solid 0.88 border rgb "#404040"
    set key at graph 0.015,0.965 left top horizontal maxrows 1 opaque box linewidth 0.55 samplen 1.0 spacing 0.75 font "Liberation Sans,9.5"
    # Show all five block means; retain CI only for the focal Uniform/Anchored comparison,
    # matching the uncertainty hierarchy used in the full-trajectory panel above.
    plot \
        zoom_file using (strcol(1) eq "DQN" ? $2-32 : 1/0):4:(14) with boxes linecolor rgb crop_dqn_color title "DQN", \
        zoom_file using (strcol(1) eq "Dueling" ? $2-16 : 1/0):4:(14) with boxes linecolor rgb crop_dueling_color title "Dueling", \
        zoom_file using (strcol(1) eq "Naive" ? $2 : 1/0):4:(14) with boxes linecolor rgb crop_naive_color title "Naive", \
        zoom_file using (strcol(1) eq "Uniform" ? $2+16 : 1/0):4:(14) with boxes linecolor rgb crop_uniform_color title "Uniform", \
        zoom_file using (strcol(1) eq "Anchored" ? $2+32 : 1/0):4:(14) with boxes linecolor rgb crop_anchored_color title "Anchored", \
        zoom_file using (strcol(1) eq "Uniform" ? $2+16 : 1/0):4:5:6 with yerrorbars linecolor rgb crop_uniform_color linewidth 1.20 pointtype -1 notitle, \
        zoom_file using (strcol(1) eq "Anchored" ? $2+32 : 1/0):4:5:6 with yerrorbars linecolor rgb crop_anchored_color linewidth 1.30 pointtype -1 notitle

    unset key
    unset multiplot
    unset output
}
