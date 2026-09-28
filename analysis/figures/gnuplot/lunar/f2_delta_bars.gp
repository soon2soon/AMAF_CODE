# Lunar F2 alternative: paired Protected-minus-Naive deltas by seed.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/lunar/f2_delta_bars.gp

load "analysis/figures/gnuplot/amaf_common.gp"

datafile = "analysis/figures/figure_data/lunar_f2_delta_bars.csv"
pdf_output = "paper_figures/gnuplot/lunar/lunar_f2_delta_bars.pdf"
png_output = "paper_figures/gnuplot/lunar/lunar_f2_delta_bars.png"

positive_bar_rgb = 0x3567A8
negative_bar_rgb = 0xC43C4A
bar_color(value) = (value >= 0 ? positive_bar_rgb : negative_bar_rgb)

set datafile separator comma
set boxwidth 0.68 absolute
set style fill solid 0.94 border linecolor rgb "#FFFFFF"
unset key
unset colorbox
unset grid
set grid ytics back linestyle 90
set xrange [0.35:10.65]
set xtics ("11" 1, "22" 2, "33" 3, "44" 4, "55" 5, \
    "66" 6, "77" 7, "88" 8, "99" 9, "111" 10)
set format y "%g"

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 6.50in,5.20in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1950,1560
        set output png_output
    }

    set multiplot

    set lmargin at screen 0.140
    set rmargin at screen 0.985
    set bmargin at screen 0.565
    set tmargin at screen 0.955
    set title "(a) Delta Last-5 by seed" font "Liberation Sans Bold,11.5" offset 0,-0.25
    unset xlabel
    set ylabel "Protected - Naive (Last-5 return)" offset 1.0,0
    set yrange [-80:430]
    set ytics 100
    set mytics 2
    set arrow 1 from graph 0, first 0 to graph 1, first 0 \
        nohead front linecolor rgb "#303030" linewidth 1.15
    plot \
        datafile using 0:2:(bar_color($2)) with boxes linecolor rgb variable notitle, \
        datafile using 0:($2 >= 0 ? $2 : 1/0):(sprintf("%+.0f", $2)) \
            with labels offset character 0,0.65 font "Liberation Sans Bold,10.5" \
            textcolor rgb "#202020" notitle, \
        datafile using 0:($2 < 0 ? $2 : 1/0):(sprintf("%+.0f", $2)) \
            with labels offset character 0,-0.65 font "Liberation Sans Bold,10.5" \
            textcolor rgb "#202020" notitle

    set lmargin at screen 0.140
    set rmargin at screen 0.985
    set bmargin at screen 0.120
    set tmargin at screen 0.490
    set title "(b) Delta Retention by seed" font "Liberation Sans Bold,11.5" offset 0,-0.25
    set xlabel "Seed" offset 0,0.25
    set ylabel "Protected - Naive retention (pp)" offset 1.0,0
    set yrange [-30:170]
    set ytics 50
    set mytics 2
    set arrow 1 from graph 0, first 0 to graph 1, first 0 \
        nohead front linecolor rgb "#303030" linewidth 1.15
    plot \
        datafile using 0:3:(bar_color($3)) with boxes linecolor rgb variable notitle, \
        datafile using 0:($3 >= 0 ? $3 : 1/0):(sprintf("%+.0f", $3)) \
            with labels offset character 0,0.65 font "Liberation Sans Bold,10.5" \
            textcolor rgb "#202020" notitle, \
        datafile using 0:($3 < 0 ? $3 : 1/0):(sprintf("%+.0f", $3)) \
            with labels offset character 0,-0.65 font "Liberation Sans Bold,10.5" \
            textcolor rgb "#202020" notitle

    unset multiplot
    unset output
}
