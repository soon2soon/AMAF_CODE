# Crop Panel B: matched-seed resource savings and yield change.
# Positive values consistently indicate benefit for Anchored over Uniform.

load "analysis/figures/gnuplot/amaf_common.gp"

data_file = "analysis/figures/figure_data/crop_panel_b_seed_resource_yield.csv"
pdf_output = "paper_figures/gnuplot/crop/crop_panel_b_resource_yield.pdf"
png_output = "paper_figures/gnuplot/crop/crop_panel_b_resource_yield.png"

set datafile separator comma
unset key
unset colorbox
unset grid
set grid ytics mytics back linestyle 90, linestyle 91
set boxwidth 0.62 relative
set style fill solid 0.90 border rgb "#303030"
set xrange [-0.65:4.65]
set xtics ("11" 0, "22" 1, "33" 2, "44" 3, "55" 4)
set mxtics 1
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#3F3F3F" linewidth 1.05

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,6.00in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1800
        set output png_output
    }

    set multiplot
    set label 100 "Positive = better for Anchored" at screen 0.985,0.982 right front font "Liberation Sans Bold,10.5" textcolor rgb "#333333"
    set lmargin at screen 0.145
    set rmargin at screen 0.985

    set bmargin at screen 0.705
    set tmargin at screen 0.950
    set title "Nitrogen saving" font "Liberation Sans Bold,11.5" offset 0,-0.25
    set ylabel "Uniform - Anchored" offset 0.8,0
    set yrange [-3:28]
    set ytics 5
    set mytics 1
    set format x ""
    unset xlabel
    plot \
        data_file using ($0-1):2:(0.62):($2 >= 0 ? 0x3567A8 : 0xC43C4A) with boxes linecolor rgb variable notitle, \
        data_file using ($0-1):($2 >= 0 ? $2+1.35 : $2-1.35):(sprintf("%+.1f", $2)) with labels font "Liberation Sans Bold,10" textcolor rgb "#222222" notitle

    set bmargin at screen 0.405
    set tmargin at screen 0.650
    set title "Irrigation saving" font "Liberation Sans Bold,11.5" offset 0,-0.25
    set ylabel "Uniform - Anchored" offset 0.8,0
    set yrange [-3:23]
    set ytics 5
    set mytics 1
    plot \
        data_file using ($0-1):3:(0.62):($3 >= 0 ? 0x3567A8 : 0xC43C4A) with boxes linecolor rgb variable notitle, \
        data_file using ($0-1):($3 >= 0 ? $3+1.15 : $3-1.15):(sprintf("%+.1f", $3)) with labels font "Liberation Sans Bold,10" textcolor rgb "#222222" notitle

    set bmargin at screen 0.115
    set tmargin at screen 0.350
    set title "Yield change" font "Liberation Sans Bold,11.5" offset 0,-0.25
    set ylabel "Anchored - Uniform" offset 0.8,0
    set yrange [-1050:750]
    set ytics 300
    set mytics 3
    set format x "%g"
    set xlabel "Seed" offset 0,0.25
    plot \
        data_file using ($0-1):4:(0.62):($4 >= 0 ? 0x3567A8 : 0xC43C4A) with boxes linecolor rgb variable notitle, \
        data_file using ($0-1):($4 >= 0 ? $4+75 : $4-75):(sprintf("%+.0f", $4)) with labels font "Liberation Sans Bold,10" textcolor rgb "#222222" notitle

    unset label 100
    unset multiplot
    unset output
}
