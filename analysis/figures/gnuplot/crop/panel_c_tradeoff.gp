# Legacy renderer retained for provenance; not used by the active production contract.
# Crop Panel C: final-100 yield-resource trade-off planes.
# Python owns canonical loading, seed selection, and centroid preparation.

load "analysis/figures/gnuplot/amaf_common.gp"

nitrogen_file = "analysis/figures/figure_data/crop_panel_c_nitrogen_tradeoff.csv"
irrigation_file = "analysis/figures/figure_data/crop_panel_c_irrigation_tradeoff.csv"
connection_file = "analysis/figures/figure_data/crop_panel_c_paired_connections.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_c_tradeoff.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_c_tradeoff.png"

set datafile separator comma
unset colorbox
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91
set xlabel font axis_label_font
set ylabel font axis_label_font

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,3.75in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1125
        set output png_output
    }

    set multiplot
    set key at screen 0.540,0.965 center horizontal maxrows 1 opaque box linewidth 0.6 samplen 1.0 spacing 0.62 font "Liberation Sans,9.6"
    set label 100 "Thin gray segments connect matched Uniform–Anchored seeds" at screen 0.540,0.905 center front font "Liberation Sans,8.8" textcolor rgb "#5A5A5A"
    set yrange [7300:10400]
    set ytics 500
    set mytics 2
    set ylabel "Yield, final 100 episodes" offset 0.8,0

    # (c1) Nitrogen. Matched segments are behind all raw points and centroids.
    set lmargin at screen 0.105
    set rmargin at screen 0.490
    set bmargin at screen 0.185
    set tmargin at screen 0.855
    set title "(c1) Nitrogen vs Yield" font "Liberation Sans Bold,11.5" offset 0,-0.15
    set xrange [195:290]
    set xtics 20
    set mxtics 2
    set xlabel "Nitrogen use, final 100 episodes" offset 0,0.25
    plot \
        connection_file using 2:6:($3-$2):($7-$6) with vectors nohead linecolor rgb "#99808080" linewidth 0.55 notitle, \
        nitrogen_file using (strcol(1) eq "DQN" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.60 notitle, \
        nitrogen_file using (strcol(1) eq "Dueling" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.64 notitle, \
        nitrogen_file using (strcol(1) eq "Naive" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.68 notitle, \
        nitrogen_file using (strcol(1) eq "Uniform" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.72 notitle, \
        nitrogen_file using (strcol(1) eq "Anchored" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.76 notitle, \
        nitrogen_file using (strcol(1) eq "DQN" && $5 == 1 ? $3 : 1/0):4 with points linestyle 71 pointsize 1.20 title "DQN", \
        nitrogen_file using (strcol(1) eq "Dueling" && $5 == 1 ? $3 : 1/0):4 with points linestyle 72 pointsize 1.24 title "Dueling", \
        nitrogen_file using (strcol(1) eq "Naive" && $5 == 1 ? $3 : 1/0):4 with points linestyle 74 pointsize 1.28 title "Naive", \
        nitrogen_file using (strcol(1) eq "Uniform" && $5 == 1 ? $3 : 1/0):4 with points linestyle 73 pointsize 1.48 title "Uniform", \
        nitrogen_file using (strcol(1) eq "Anchored" && $5 == 1 ? $3 : 1/0):4 with points linestyle 75 pointsize 1.66 title "Anchored"

    # (c2) Irrigation; use the same yield range and visual hierarchy.
    unset key
    set lmargin at screen 0.600
    set rmargin at screen 0.985
    set bmargin at screen 0.185
    set tmargin at screen 0.855
    set title "(c2) Irrigation vs Yield" font "Liberation Sans Bold,11.5" offset 0,-0.15
    set xrange [185:280]
    set xtics 20
    set mxtics 2
    set xlabel "Irrigation use, final 100 episodes" offset 0,0.25
    set ylabel ""
    plot \
        connection_file using 4:6:($5-$4):($7-$6) with vectors nohead linecolor rgb "#99808080" linewidth 0.55 notitle, \
        irrigation_file using (strcol(1) eq "DQN" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.60 notitle, \
        irrigation_file using (strcol(1) eq "Dueling" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.64 notitle, \
        irrigation_file using (strcol(1) eq "Naive" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.68 notitle, \
        irrigation_file using (strcol(1) eq "Uniform" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.72 notitle, \
        irrigation_file using (strcol(1) eq "Anchored" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.76 notitle, \
        irrigation_file using (strcol(1) eq "DQN" && $5 == 1 ? $3 : 1/0):4 with points linestyle 71 pointsize 1.20 notitle, \
        irrigation_file using (strcol(1) eq "Dueling" && $5 == 1 ? $3 : 1/0):4 with points linestyle 72 pointsize 1.24 notitle, \
        irrigation_file using (strcol(1) eq "Naive" && $5 == 1 ? $3 : 1/0):4 with points linestyle 74 pointsize 1.28 notitle, \
        irrigation_file using (strcol(1) eq "Uniform" && $5 == 1 ? $3 : 1/0):4 with points linestyle 73 pointsize 1.48 notitle, \
        irrigation_file using (strcol(1) eq "Anchored" && $5 == 1 ? $3 : 1/0):4 with points linestyle 75 pointsize 1.66 notitle

    unset label 100
    unset multiplot
    unset output
}
