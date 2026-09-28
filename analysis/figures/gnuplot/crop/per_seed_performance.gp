# Crop reviewer evidence: five independently rendered seed trajectories.
# Python owns the 100-episode aggregation and canonical anchor checks;
# gnuplot filters the renderer-ready long-format CSV and renders only.

load "analysis/figures/gnuplot/amaf_common.gp"

data_file = "analysis/figures/figure_data/crop_per_seed_learning.csv"
output_dir = "paper_figures/gnuplot/crop"
seed_list = "11 22 33 44 55"

set datafile separator comma
unset colorbox
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91
set xlabel font axis_label_font
set ylabel font axis_label_font

do for [seed_index=1:words(seed_list)] {
    seed_value = int(word(seed_list, seed_index))

    do for [render_index=1:2] {
        if (render_index == 1) {
            set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,5.20in
            set output sprintf("%s/crop_performance_seed%d.pdf", output_dir, seed_value)
        } else {
            set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1560
            set output sprintf("%s/crop_performance_seed%d.png", output_dir, seed_value)
        }

        set multiplot

        set key at graph 0.985,0.035 right bottom horizontal maxcols 3 opaque box linewidth 0.6 samplen 1.4 spacing 0.8 font "Liberation Sans,10"
        set lmargin at screen 0.125
        set rmargin at screen 0.985
        set bmargin at screen 0.390
        set tmargin at screen 0.925
        set title sprintf("Seed %d — full training trajectory (Episodes 1–3000)", seed_value) font "Liberation Sans Bold,11.5" offset 0,-0.2
        set xrange [0:3000]
        set yrange [-8300:1350]
        set xtics 500
        set mxtics 2
        set ytics 1000
        set mytics 2
        set xlabel ""
        set ylabel "Train return (100-episode block)" offset 1.0,0
        set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#424242" linewidth 1.00
        set arrow 2 from first 2000, graph 0 to first 2000, graph 1 nohead back linecolor rgb "#8A8A8A" linewidth 0.85 dashtype 2
        plot \
            data_file using (int($1) == seed_value && strcol(3) eq "DQN" ? $2 : 1/0):4 with linespoints linestyle 71 pointinterval 2 pointsize 0.60 title "DQN", \
            data_file using (int($1) == seed_value && strcol(3) eq "Dueling" ? $2 : 1/0):4 with linespoints linestyle 72 pointinterval 2 pointsize 0.64 title "Dueling", \
            data_file using (int($1) == seed_value && strcol(3) eq "Uniform" ? $2 : 1/0):4 with linespoints linestyle 73 pointinterval 2 pointsize 0.72 title "Uniform", \
            data_file using (int($1) == seed_value && strcol(3) eq "Naive" ? $2 : 1/0):4 with linespoints linestyle 74 pointinterval 2 pointsize 0.70 title "Naive", \
            data_file using (int($1) == seed_value && strcol(3) eq "Anchored" ? $2 : 1/0):4 with linespoints linestyle 75 pointinterval 2 pointsize 0.78 title "Anchored"

        unset key
        unset arrow 1
        unset arrow 2
        set lmargin at screen 0.125
        set rmargin at screen 0.985
        set bmargin at screen 0.105
        set tmargin at screen 0.300
        set title "Late-stage trajectory (Episodes 2001–3000)" font "Liberation Sans Bold,11.5" offset 0,-0.15
        set xrange [2000:3000]
        set yrange [250:1250]
        set xtics 200
        set mxtics 2
        set ytics 200
        set mytics 2
        set xlabel "Episode block end" offset 0,0.25
        set ylabel "Train return" offset 1.0,0
        plot \
            data_file using (int($1) == seed_value && $5 == 1 && strcol(3) eq "DQN" ? $2 : 1/0):4 with linespoints linestyle 71 pointinterval 1 pointsize 0.64 notitle, \
            data_file using (int($1) == seed_value && $5 == 1 && strcol(3) eq "Dueling" ? $2 : 1/0):4 with linespoints linestyle 72 pointinterval 1 pointsize 0.68 notitle, \
            data_file using (int($1) == seed_value && $5 == 1 && strcol(3) eq "Uniform" ? $2 : 1/0):4 with linespoints linestyle 73 pointinterval 1 pointsize 0.76 notitle, \
            data_file using (int($1) == seed_value && $5 == 1 && strcol(3) eq "Naive" ? $2 : 1/0):4 with linespoints linestyle 74 pointinterval 1 pointsize 0.74 notitle, \
            data_file using (int($1) == seed_value && $5 == 1 && strcol(3) eq "Anchored" ? $2 : 1/0):4 with linespoints linestyle 75 pointinterval 1 pointsize 0.82 notitle

        unset multiplot
        unset output
    }
}
