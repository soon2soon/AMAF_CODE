# Crop Panel C2: final-100 irrigation-yield trade-off.

load "analysis/figures/gnuplot/amaf_common.gp"

data_file = "analysis/figures/figure_data/crop_panel_c_irrigation_tradeoff.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_c2_irrigation_tradeoff.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_c2_irrigation_tradeoff.png"

set datafile separator comma
unset colorbox
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91
set xrange [185:280]
set yrange [7300:10400]
set xtics 20
set mxtics 2
set ytics 500
set mytics 2
set xlabel "Irrigation use, final 100 episodes" offset 0,0.25
set ylabel "Yield, final 100 episodes" offset 0.8,0
set label 1 "Lower resource use is better" at graph 0.02,0.965 left front font "Liberation Sans,9.5" textcolor rgb "#555555"

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 4.60in,3.65in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1380,1095
        set output png_output
    }
    set lmargin at screen 0.155
    set rmargin at screen 0.985
    set bmargin at screen 0.175
    set tmargin at screen 0.855
    set key at screen 0.550,0.975 center horizontal maxrows 1 opaque box linewidth 0.6 samplen 1.0 spacing 0.55 font "Liberation Sans,9.4"
    plot \
        data_file using (strcol(1) eq "DQN" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dqn_raw_color pointtype 6 pointsize 0.62 notitle, \
        data_file using (strcol(1) eq "Dueling" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_dueling_raw_color pointtype 8 pointsize 0.65 notitle, \
        data_file using (strcol(1) eq "Naive" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_naive_raw_color pointtype 2 pointsize 0.70 notitle, \
        data_file using (strcol(1) eq "Uniform" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_uniform_raw_color pointtype 4 pointsize 0.75 notitle, \
        data_file using (strcol(1) eq "Anchored" && $5 == 0 ? $3 : 1/0):4 with points linecolor rgb crop_anchored_raw_color pointtype 13 pointsize 0.76 notitle, \
        data_file using (strcol(1) eq "DQN" && $5 == 1 ? $3 : 1/0):4 with points linestyle 71 pointsize 1.22 title "DQN", \
        data_file using (strcol(1) eq "Dueling" && $5 == 1 ? $3 : 1/0):4 with points linestyle 72 pointsize 1.24 title "Dueling", \
        data_file using (strcol(1) eq "Uniform" && $5 == 1 ? $3 : 1/0):4 with points linestyle 73 pointsize 1.38 title "Uniform", \
        data_file using (strcol(1) eq "Naive" && $5 == 1 ? $3 : 1/0):4 with points linestyle 74 pointsize 1.28 title "Naive", \
        data_file using (strcol(1) eq "Anchored" && $5 == 1 ? $3 : 1/0):4 with points linestyle 75 pointsize 1.48 title "Anchored"
    unset output
}
