# Crop reviewer view: complete context, post-catastrophic absolute performance,
# and the aligned final-100 outcome profile.
# Python owns block aggregation and bootstrap statistics.

load "analysis/figures/gnuplot/amaf_common.gp"

main_file = "analysis/figures/figure_data/crop_panel_a_main_curve.csv"
profile_file = "analysis/figures/figure_data/crop_panel_c_outcome_profile.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_a_reviewer_three_view.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_a_reviewer_three_view.png"

# Focal methods dominate; conventional/secondary methods remain visible.
set style line 81 linecolor rgb crop_dqn_raw_color linewidth 0.80 dashtype 1 pointtype 6 pointsize 0.48
set style line 82 linecolor rgb crop_dueling_raw_color linewidth 0.85 dashtype 1 pointtype 8 pointsize 0.52
set style line 83 linecolor rgb crop_uniform_color linewidth 1.80 dashtype 1 pointtype 4 pointsize 0.76
set style line 84 linecolor rgb crop_naive_raw_color linewidth 0.90 dashtype 1 pointtype 2 pointsize 0.60
set style line 85 linecolor rgb crop_anchored_color linewidth 2.00 dashtype 1 pointtype 13 pointsize 0.84

set datafile separator comma
unset colorbox
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91
set xlabel font axis_label_font
set ylabel font axis_label_font

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,8.30in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,2490
        set output png_output
    }

    set multiplot

    # (a) Complete absolute-return context.
    set key at graph 0.985,0.035 right bottom horizontal maxcols 3 opaque box linewidth 0.6 samplen 1.4 spacing 0.8 font "Liberation Sans,10"
    set lmargin at screen 0.125
    set rmargin at screen 0.985
    set bmargin at screen 0.680
    set tmargin at screen 0.955
    set title "(a) Full-training overview (Episodes 1–3000)" font "Liberation Sans Bold,11.5" offset 0,-0.2
    set xrange [0:3000]
    set yrange [-8300:1350]
    set xtics 500
    set mxtics 2
    set ytics 1000
    set mytics 2
    set format y "%g"
    set xlabel ""
    set ylabel "Mean train return (100-episode block)" offset 1.0,0
    set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#424242" linewidth 1.00
    plot \
        main_file using (strcol(1) eq "Uniform" ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_uniform_color fillstyle transparent solid 0.07 noborder notitle, \
        main_file using (strcol(1) eq "Anchored" ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_anchored_color fillstyle transparent solid 0.07 noborder notitle, \
        main_file using (strcol(1) eq "DQN" ? $2 : 1/0):4 with linespoints linestyle 81 pointinterval 3 title "DQN", \
        main_file using (strcol(1) eq "Dueling" ? $2 : 1/0):4 with linespoints linestyle 82 pointinterval 3 title "Dueling", \
        main_file using (strcol(1) eq "Naive" ? $2 : 1/0):4 with linespoints linestyle 84 pointinterval 3 title "Naive", \
        main_file using (strcol(1) eq "Uniform" ? $2 : 1/0):4 with linespoints linestyle 83 pointinterval 2 title "Uniform", \
        main_file using (strcol(1) eq "Anchored" ? $2 : 1/0):4 with linespoints linestyle 85 pointinterval 2 title "Anchored"

    # (b) Fixed post-catastrophic phase boundary: first included block ends at 1600.
    unset key
    unset arrow 1
    set lmargin at screen 0.125
    set rmargin at screen 0.985
    set bmargin at screen 0.400
    set tmargin at screen 0.625
    set title "(b) Absolute performance after the catastrophic phase (Episodes 1501–3000)" font "Liberation Sans Bold,11.5" offset 0,-0.15
    set xrange [1500:3000]
    set yrange [0:1200]
    set xtics 250
    set mxtics 2
    set ytics 200
    set mytics 2
    set format y "%g"
    set xlabel "Episode block end" offset 0,0.25
    set ylabel "Mean train return" offset 1.0,0
    plot \
        main_file using (strcol(1) eq "Uniform" && $2 >= 1600 ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_uniform_color fillstyle transparent solid 0.09 noborder notitle, \
        main_file using (strcol(1) eq "Anchored" && $2 >= 1600 ? $2 : 1/0):5:6 with filledcurves linecolor rgb crop_anchored_color fillstyle transparent solid 0.09 noborder notitle, \
        main_file using (strcol(1) eq "DQN" && $2 >= 1600 ? $2 : 1/0):4 with linespoints linestyle 81 pointinterval 2 notitle, \
        main_file using (strcol(1) eq "Dueling" && $2 >= 1600 ? $2 : 1/0):4 with linespoints linestyle 82 pointinterval 2 notitle, \
        main_file using (strcol(1) eq "Naive" && $2 >= 1600 ? $2 : 1/0):4 with linespoints linestyle 84 pointinterval 2 notitle, \
        main_file using (strcol(1) eq "Uniform" && $2 >= 1600 ? $2 : 1/0):4 with linespoints linestyle 83 pointinterval 2 notitle, \
        main_file using (strcol(1) eq "Anchored" && $2 >= 1600 ? $2 : 1/0):4 with linespoints linestyle 85 pointinterval 2 notitle

    # (c) Five-method final-100 outcome profile in native units.
    unset key
    set label 100 "(c) Final-100 outcome profile" at screen 0.125,0.350 left front font "Liberation Sans Bold,11.5" textcolor rgb "#202020"
    set label 101 "Small: five seeds    Large: method mean" at screen 0.985,0.350 right front font "Liberation Sans,9.0" textcolor rgb "#4A4A4A"
    set yrange [0.45:5.55]
    set ytics ("DQN" 5, "Dueling" 4, "Uniform" 3, "Naive" 2, "Anchored" 1)
    set mytics 1
    set ylabel ""

    # (c1) Yield: higher is better.
    set lmargin at screen 0.130
    set rmargin at screen 0.405
    set bmargin at screen 0.070
    set tmargin at screen 0.295
    set title "Yield — higher is better" font "Liberation Sans Bold,10.5" offset 0,-0.10
    set xrange [7300:10400]
    set xtics 500
    set mxtics 2
    set xlabel "Yield, final 100 episodes" offset 0,0.25
    set format y "%g"
    plot \
        profile_file using (strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "yield" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "yield" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    # (c2) Nitrogen: lower is better.
    set lmargin at screen 0.455
    set rmargin at screen 0.710
    set bmargin at screen 0.070
    set tmargin at screen 0.295
    set title "Nitrogen — lower is better" font "Liberation Sans Bold,10.5" offset 0,-0.10
    set xrange [195:290]
    set xtics 20
    set mxtics 2
    set xlabel "Nitrogen use, final 100 episodes" offset 0,0.25
    set format y ""
    plot \
        profile_file using (strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "nitrogen" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "nitrogen" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    # (c3) Irrigation: lower is better.
    set lmargin at screen 0.760
    set rmargin at screen 0.985
    set bmargin at screen 0.070
    set tmargin at screen 0.295
    set title "Irrigation — lower is better" font "Liberation Sans Bold,10.5" offset 0,-0.10
    set xrange [185:280]
    set xtics 20
    set mxtics 2
    set xlabel "Irrigation use, final 100 episodes" offset 0,0.25
    set format y ""
    plot \
        profile_file using (strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with lines linecolor rgb "#777777" linewidth 0.90 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.54 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.57 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.61 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.64 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "irrigation" && $6 == 0 ? $5 : 1/0):2 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.67 notitle, \
        profile_file using (strcol(1) eq "DQN" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 71 pointsize 1.06 notitle, \
        profile_file using (strcol(1) eq "Dueling" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 72 pointsize 1.10 notitle, \
        profile_file using (strcol(1) eq "Naive" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 74 pointsize 1.14 notitle, \
        profile_file using (strcol(1) eq "Uniform" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 73 pointsize 1.42 notitle, \
        profile_file using (strcol(1) eq "Anchored" && strcol(3) eq "irrigation" && $6 == 1 ? $5 : 1/0):2 with points linestyle 75 pointsize 1.60 notitle

    unset label 100
    unset label 101
    unset key
    unset multiplot
    unset output
}
