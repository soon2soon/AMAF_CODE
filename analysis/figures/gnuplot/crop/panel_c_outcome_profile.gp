# Crop Panel C: aligned final-100 outcome profile in native units.
# Python owns canonical loading, seed inclusion, and method means.

load "analysis/figures/gnuplot/amaf_common.gp"

data_file = "analysis/figures/figure_data/crop_panel_c_outcome_profile.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_c_outcome_profile.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_c_outcome_profile.png"

set datafile separator comma
unset colorbox
unset key
set grid xtics ytics mxtics back linestyle 90, linestyle 91
set xrange restore
set yrange [0.45:5.55]
set ytics ("DQN" 5, "Dueling" 4, "Uniform" 3, "Naive" 2, "Anchored" 1)
set mytics 1
set xlabel font axis_label_font

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,3.65in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1095
        set output png_output
    }

    set multiplot
    set label 100 "Small markers: seeds 11, 22, 33, 44, 55    Large markers: method mean" at screen 0.985,0.955 right front font "Liberation Sans,9.2" textcolor rgb "#4A4A4A"

    # (c1) Yield: higher is better.
    set lmargin at screen 0.130
    set rmargin at screen 0.405
    set bmargin at screen 0.160
    set tmargin at screen 0.885
    set title "(c1) Yield — higher is better" font "Liberation Sans Bold,11.2" offset 0,-0.15
    set xrange [7300:10400]
    set xtics 500
    set mxtics 2
    set xlabel "Yield, final 100 episodes" offset 0,0.25
    set format y "%g"
    plot \
        data_file using (strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    # (c2) Nitrogen: lower is better.
    set lmargin at screen 0.455
    set rmargin at screen 0.710
    set bmargin at screen 0.160
    set tmargin at screen 0.885
    set title "(c2) Nitrogen — lower is better" font "Liberation Sans Bold,11.2" offset 0,-0.15
    set xrange [195:290]
    set xtics 20
    set mxtics 2
    set xlabel "Nitrogen use, final 100 episodes" offset 0,0.25
    set format y ""
    plot \
        data_file using (strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    # (c3) Irrigation: lower is better.
    set lmargin at screen 0.760
    set rmargin at screen 0.985
    set bmargin at screen 0.160
    set tmargin at screen 0.885
    set title "(c3) Irrigation — lower is better" font "Liberation Sans Bold,11.2" offset 0,-0.15
    set xrange [185:280]
    set xtics 20
    set mxtics 2
    set xlabel "Irrigation use, final 100 episodes" offset 0,0.25
    set format y ""
    plot \
        data_file using (strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        data_file using (strcol(1) eq "DQN" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        data_file using (strcol(1) eq "Dueling" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        data_file using (strcol(1) eq "Naive" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        data_file using (strcol(1) eq "Uniform" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        data_file using (strcol(1) eq "Anchored" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    unset label 100
    unset multiplot
    unset output
}
