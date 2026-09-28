# Lunar F2: seed-level failure and rescue matrix.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/lunar/f2_failure_rescue.gp

load "analysis/figures/gnuplot/amaf_common.gp"

datafile = "analysis/figures/figure_data/lunar_f2_failure_rescue.csv"
pdf_output = "paper_figures/gnuplot/lunar/lunar_f2_failure_rescue.pdf"
png_output = "paper_figures/gnuplot/lunar/lunar_f2_failure_rescue.png"

set datafile separator comma
unset grid
unset colorbox
set style fill solid 1.0 border linecolor rgb "#FFFFFF"

# Raw columns are neutral. Only Delta columns carry semantic red/blue color.
lunar_delta_negative_rgb = 0xC43C4A
lunar_delta_neutral_rgb = 0xF5F5F3
lunar_delta_positive_rgb = 0x3567A8
lunar_raw_cell_rgb = 0xF1F2F4

clip01(x) = (x < 0 ? 0 : (x > 1 ? 1 : x))
rgb24(r, g, b) = int(r) * 65536 + int(g) * 256 + int(b)
red24(c) = int(c / 65536)
green24(c) = int(c / 256) % 256
blue24(c) = int(c) % 256
mixch(a, b, t) = int(a + (b - a) * clip01(t) + 0.5)
blend_rgb(c0, c1, t) = rgb24( \
    mixch(red24(c0), red24(c1), t), \
    mixch(green24(c0), green24(c1), t), \
    mixch(blue24(c0), blue24(c1), t))
# The exported color_score is value/max(abs(value)) per metric. The two Delta
# columns therefore retain independent, symmetric, zero-centered scales.
delta_strength(z) = clip01(abs(z)) ** 0.65
delta_color(z) = (z < 0 \
    ? blend_rgb(lunar_delta_neutral_rgb, lunar_delta_negative_rgb, delta_strength(z)) \
    : blend_rgb(lunar_delta_neutral_rgb, lunar_delta_positive_rgb, delta_strength(z)))
delta_text_color(z) = (delta_strength(z) >= 0.72 ? 0xFFFFFF : 0x181818)
plot_x(x) = x + (x > 3 ? 0.20 : 0.0)

set xrange [0.45:6.75]
set yrange [0.5:10.5]
set xtics ( \
    "Naive\nLast-5" 1, \
    "Protected\nLast-5" 2, \
    "{/:Bold Delta}\n{/:Bold Last-5}" 3, \
    "Naive\nRetention" 4.2, \
    "Protected\nRetention" 5.2, \
    "{/:Bold Delta}\n{/:Bold Retention (pp)}" 6.2 \
)
set ytics ("111" 1, "99" 2, "88" 3, "77" 4, "66" 5, "55" 6, "44" 7, "33" 8, "22" 9, "11" 10)
set xlabel ""
set ylabel "Seed" offset 1.2,0
unset key

set lmargin at screen 0.095
set rmargin at screen 0.990
set bmargin at screen 0.160
set tmargin at screen 0.980

set object 1 rect from first 0.50, graph 0 to first 3.50, graph 1 behind \
    fillcolor rgb "#FCFCFC" fillstyle solid 1.0 border linecolor rgb "#707070" linewidth 0.8
set object 2 rect from first 3.70, graph 0 to first 6.70, graph 1 behind \
    fillcolor rgb "#FCFCFC" fillstyle solid 1.0 border linecolor rgb "#707070" linewidth 0.8

set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 6.80in,4.10in
set output pdf_output
plot \
    datafile using (($4 == 3 || $4 == 6) ? 1/0 : plot_x($4)):2:($8 + ($4 > 3 ? 0.20 : 0)):($9 + ($4 > 3 ? 0.20 : 0)):10:11:(lunar_raw_cell_rgb) \
        with boxxyerrorbars linecolor rgb variable notitle, \
    datafile using (($4 == 3 || $4 == 6) ? plot_x($4) : 1/0):2:($8 + ($4 > 3 ? 0.20 : 0)):($9 + ($4 > 3 ? 0.20 : 0)):10:11:(delta_color($7)) \
        with boxxyerrorbars linecolor rgb variable notitle, \
    datafile using (($4 == 3 || $4 == 6) ? 1/0 : plot_x($4)):2:6 \
        with labels font "Liberation Sans,11" textcolor rgb "#202020" notitle, \
    datafile using (($4 == 3 || $4 == 6) ? plot_x($4) : 1/0):2:6:(delta_text_color($7)) \
        with labels font "Liberation Sans Bold,11" textcolor rgb variable notitle
unset output

set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2040,1230
set output png_output
replot
unset output
