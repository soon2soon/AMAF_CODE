# Discussion Figure D2: seed-level claim boundaries and trade-offs.
# Run from the repository root after:
#   python -m analysis.figures.export_discussion_figures

load "analysis/figures/gnuplot/amaf_common.gp"

lunar_file = "analysis/figures/data/discussion/discussion_d2_lunar_loo.tsv"
crop_file = "analysis/figures/data/discussion/discussion_d2_crop_tradeoff.tsv"
halfcheetah_file = "analysis/figures/data/discussion/discussion_d2_halfcheetah_boundary.tsv"
pdf_output = "paper_figures/gnuplot/discussion/discussion_claim_boundaries.pdf"
png_output = "paper_figures/gnuplot/discussion/discussion_claim_boundaries.png"

set datafile separator "\t"

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,5.60in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,1680
        set output png_output
    }

    set multiplot
    set grid xtics ytics back linestyle 90

    # (a) Lunar Protected - Naive leave-one-seed-out stability.
    set lmargin at screen 0.100
    set rmargin at screen 0.485
    set bmargin at screen 0.595
    set tmargin at screen 0.940
    set title "(a) Lunar: leave-one-seed-out Last-5 effect" font "Liberation Sans Bold,10.5" offset 0,-0.1
    set xrange [0.5:10.5]
    set yrange [-5:110]
    set xtics ("11" 1, "22" 2, "33" 3, "44" 4, "55" 5, "66" 6, "77" 7, "88" 8, "99" 9, "111" 10) font "Liberation Sans,8.5"
    set ytics 20
    set mytics 2
    set xlabel "Excluded seed" offset 0,0.25
    set ylabel "LOO mean delta\nProtected - Naive" offset 1.0,0
    set key left bottom opaque box linewidth 0.5 samplen 1.5 spacing 0.9 font "Liberation Sans,8.5"
    set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#555555" linewidth 0.9
    plot \
        lunar_file using 2:6 with lines linecolor rgb "#555555" linewidth 1.1 dashtype 2 title "Full-sample mean", \
        lunar_file using 2:3 with points pointtype 6 pointsize 0.95 linecolor rgb benefit_positive_color title "LOO mean"

    # (b1) Crop nitrogen saving versus yield change.
    unset key
    set lmargin at screen 0.595
    set rmargin at screen 0.960
    set bmargin at screen 0.595
    set tmargin at screen 0.940
    set title "(b1) Crop: nitrogen saving and yield change" font "Liberation Sans Bold,10.5" offset 0,-0.1
    set xrange [-950:650]
    set yrange [-2:25]
    set xtics 400
    set mxtics 2
    set ytics 5
    set mytics 1
    set xlabel "Yield delta (Anchored - Uniform)" offset 0,0.25
    set ylabel "Nitrogen saving\n(Uniform - Anchored)" offset 1.0,0
    unset arrow 1
    set arrow 2 from first 0, graph 0 to first 0, graph 1 nohead front linecolor rgb "#555555" linewidth 0.9
    set arrow 3 from graph 0, first 0 to graph 1, first 0 nohead front linecolor rgb "#555555" linewidth 0.9
    plot \
        crop_file using 2:3 with points pointtype 6 pointsize 1.05 linecolor rgb benefit_positive_color notitle, \
        crop_file using 2:3:(sprintf("%d", $1)) with labels offset 0.7,0.5 font "Liberation Sans,8.2" textcolor rgb "#333333" notitle

    # (b2) Crop irrigation saving versus the same yield change.
    set lmargin at screen 0.100
    set rmargin at screen 0.485
    set bmargin at screen 0.115
    set tmargin at screen 0.460
    set title "(b2) Crop: irrigation saving and yield change" font "Liberation Sans Bold,10.5" offset 0,-0.1
    set xlabel "Yield delta (Anchored - Uniform)" offset 0,0.25
    set ylabel "Irrigation saving\n(Uniform - Anchored)" offset 1.0,0
    plot \
        crop_file using 2:4 with points pointtype 8 pointsize 1.05 linecolor rgb benefit_positive_color notitle, \
        crop_file using 2:4:(sprintf("%d", $1)) with labels offset 0.7,0.5 font "Liberation Sans,8.2" textcolor rgb "#333333" notitle

    # (c) HalfCheetah terminal effect and 1M-to-1.5M relative gain are distinct.
    set lmargin at screen 0.595
    set rmargin at screen 0.960
    set bmargin at screen 0.115
    set tmargin at screen 0.460
    set title "(c) HalfCheetah: terminal versus late-stage gain" font "Liberation Sans Bold,10.5" offset 0,-0.1
    set xrange [-4200:2000]
    set yrange [-700:1900]
    set xtics 1000
    set mxtics 2
    set ytics 500
    set mytics 2
    set xlabel "Terminal delta at 1.5M (AMAF - TD3)" offset 0,0.25
    set ylabel "Gain delta, 1M to 1.5M\n(AMAF - TD3)" offset 1.0,0
    set key left top opaque box linewidth 0.5 samplen 1.2 spacing 0.9 font "Liberation Sans,8.5"
    plot \
        halfcheetah_file using (strcol(1) eq "original5" ? $3 : 1/0):4 \
            with points pointtype 6 pointsize 1.05 linecolor rgb amaf_color title "Original 5", \
        halfcheetah_file using (strcol(1) eq "replication5" ? $3 : 1/0):4 \
            with points pointtype 8 pointsize 1.08 linecolor rgb amaf_color title "Replication 5", \
        halfcheetah_file using ($2 == 77 ? $3 : 1/0):4:("seed 77") \
            with labels offset 1.0,0.7 left font "Liberation Sans Bold,8.5" textcolor rgb benefit_negative_color notitle

    unset arrow 2
    unset arrow 3
    unset key
    unset multiplot
    unset output
}
