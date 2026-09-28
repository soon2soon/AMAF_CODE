# Discussion Figure D1: checkpoint-time routing diagnostics.
# Run from the repository root after:
#   python -m analysis.figures.export_discussion_figures

load "analysis/figures/gnuplot/amaf_common.gp"

heatmap_file = "analysis/figures/data/discussion/discussion_d1_head_heatmap.tsv"
switching_file = "analysis/figures/data/discussion/discussion_d1_switching.tsv"
entropy_summary_file = "analysis/figures/data/discussion/routing_entropy_phase_quantiles.csv"
pdf_output = "paper_figures/gnuplot/discussion/discussion_routing_mechanism.pdf"
png_output = "paper_figures/gnuplot/discussion/discussion_routing_mechanism.png"

set datafile separator "\t"

# Absolute head weights are preserved in the TSV; rendering saturates values
# above 0.5 so the common 0-to-0.5 scale retains visible contrast.
set palette defined (0 td3_color, 0.5 "white", 1 amaf_color)
set cbrange [0:0.5]
set cbtics 0.25
set format cb "%.2f"

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 7.20in,6.80in
        set output pdf_output
    } else {
        set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 2160,2040
        set output png_output
    }

    set multiplot
    unset key
    unset grid
    unset colorbox
    set xrange [0:1]
    set xtics 0.25
    set mxtics 1
    unset mytics
    set xlabel font "Liberation Sans,10.5"
    set ylabel font "Liberation Sans,10.5"

    set label 101 "(a1) Lunar Protected" at screen 0.210,0.968 center front font "Liberation Sans Bold,11"
    set label 102 "(a2) Crop Anchored" at screen 0.520,0.968 center front font "Liberation Sans Bold,11"
    set label 103 "(a3) HalfCheetah AMAF" at screen 0.790,0.968 center front font "Liberation Sans Bold,11"

    # (a1-i) Lunar raw router.
    set lmargin at screen 0.075
    set rmargin at screen 0.345
    set bmargin at screen 0.735
    set tmargin at screen 0.915
    set yrange [-0.55:3.55]
    set ytics ("H0" 0, "H1" 1, "H2" 2, "H3" 3)
    set format x ""
    set format y "%s"
    unset xlabel
    set ylabel "Head index" offset 1.0,0
    set title "Raw router (across-seed mean)" font "Liberation Sans Bold,9.5" offset 0,-0.15
    plot heatmap_file using \
        (strcol(1) eq "Lunar" && strcol(15) eq "raw_router" ? $4 : 1/0):5:11:12:13:14:($7 > 0.5 ? 0.5 : $7) \
        with boxxyerror linecolor palette fillstyle solid 1.0 noborder notitle

    # (a1-ii) Lunar effective fusion after reference preservation.
    set bmargin at screen 0.500
    set tmargin at screen 0.680
    set format x "%g"
    set xlabel "Normalized training progress" offset 0,0.2
    set ylabel "Head index" offset 1.0,0
    set title "Effective gate mean  (Uniform = 0.25/head)" font "Liberation Sans Bold,9.2" offset 0,-0.15
    plot heatmap_file using \
        (strcol(1) eq "Lunar" && strcol(15) eq "effective_gate" ? $4 : 1/0):5:11:12:13:14:($7 > 0.5 ? 0.5 : $7) \
        with boxxyerror linecolor palette fillstyle solid 1.0 noborder notitle

    # (a2-i) Crop raw router.
    set lmargin at screen 0.390
    set rmargin at screen 0.650
    set bmargin at screen 0.735
    set tmargin at screen 0.915
    set yrange [-0.55:2.55]
    set ytics ("H0" 0, "H1" 1, "H2" 2)
    set format x ""
    unset xlabel
    set ylabel ""
    set title "Raw router (across-seed mean)" font "Liberation Sans Bold,9.5" offset 0,-0.15
    plot heatmap_file using \
        (strcol(1) eq "Crop" && strcol(15) eq "raw_router" ? $4 : 1/0):5:11:12:13:14:($7 > 0.5 ? 0.5 : $7) \
        with boxxyerror linecolor palette fillstyle solid 1.0 noborder notitle

    # (a2-ii) Crop effective fusion after anchoring.
    set bmargin at screen 0.500
    set tmargin at screen 0.680
    set format x "%g"
    set xlabel "Normalized training progress" offset 0,0.2
    set title "Effective gate mean  (Uniform = 1/3 per head)" font "Liberation Sans Bold,9.2" offset 0,-0.15
    plot heatmap_file using \
        (strcol(1) eq "Crop" && strcol(15) eq "effective_gate" ? $4 : 1/0):5:11:12:13:14:($7 > 0.5 ? 0.5 : $7) \
        with boxxyerror linecolor palette fillstyle solid 1.0 noborder notitle

    # (a3) Only the recorded conditional residual-head distribution is available.
    set lmargin at screen 0.695
    set rmargin at screen 0.890
    set bmargin at screen 0.570
    set tmargin at screen 0.865
    set yrange [-0.55:3.55]
    set ytics ("H0" 0, "H1" 1, "H2" 2, "H3" 3)
    set format x "%g"
    set xlabel "Normalized training progress" offset 0,0.2
    set ylabel ""
    set title "Conditional residual-head weight (mean)" font "Liberation Sans Bold,9.2" offset 0,-0.15
    set colorbox vertical user origin 0.910,0.590 size 0.012,0.245
    set cblabel "Head weight" font "Liberation Sans,9.5" offset 0.2,0
    plot heatmap_file using \
        (strcol(1) eq "HalfCheetah" && strcol(15) eq "effective_gate" ? $4 : 1/0):5:11:12:13:14:($7 > 0.5 ? 0.5 : $7) \
        with boxxyerror linecolor palette fillstyle solid 1.0 noborder notitle

    unset label 101
    unset label 102
    unset label 103
    unset colorbox

    # (b) Nested checkpoint-time entropy quantile intervals.
    set datafile separator comma
    set border 15 back linewidth 1.05 linecolor rgb "#262626"
    unset key
    set grid xtics back linestyle 90
    set lmargin at screen 0.110
    set rmargin at screen 0.575
    set bmargin at screen 0.105
    set tmargin at screen 0.375
    set xrange [0:1]
    set yrange [0.5:11.6]
    set xtics 0.2 font "Liberation Sans,8.0"
    set mxtics 2
    set ytics (\
        "Early" 10.4, "Middle" 9.4, "Late" 8.4, \
        "Early" 6.8, "Middle" 5.8, "Late" 4.8, \
        "Early" 3.2, "Middle" 2.2, "Late" 1.2) font "Liberation Sans,7.8"
    unset mytics
    set format x "%.1f"
    set format y "%s"
    set xlabel "Normalized routing entropy   (0 = concentrated; 1 = diffuse)" offset 0,0.15
    set ylabel ""
    unset title
    set label 201 "(b) Routing concentration across training phases" at screen 0.090,0.405 left front font "Liberation Sans Bold,10.5"
    set label 202 "Lunar Protected (raw router)" at first 0.015,11.15 left front font "Liberation Sans Bold,8.6"
    set label 203 "Crop Anchored (raw router)" at first 0.015,7.55 left front font "Liberation Sans Bold,8.6"
    set label 204 "HalfCheetah AMAF (conditional residual heads)" at first 0.015,3.95 left front font "Liberation Sans Bold,8.6"

    # Each row layers 5--95%, 10--90%, and 25--75% intervals, then the median.
    plot \
        entropy_summary_file using (strcol(1) eq "Lunar" ? $7 : 1/0):(11.4-$5):($13-$7):(0) with vectors nohead linecolor rgb "#BFD7D0" linewidth 1.0 notitle, \
        entropy_summary_file using (strcol(1) eq "Lunar" ? $8 : 1/0):(11.4-$5):($12-$8):(0) with vectors nohead linecolor rgb "#79A696" linewidth 2.5 notitle, \
        entropy_summary_file using (strcol(1) eq "Lunar" ? $9 : 1/0):(11.4-$5):($11-$9):(0) with vectors nohead linecolor rgb lunar_protected_color linewidth 5.0 notitle, \
        entropy_summary_file using (strcol(1) eq "Lunar" ? $10 : 1/0):(11.4-$5) with points pointtype 7 pointsize 0.85 linecolor rgb lunar_protected_color notitle, \
        entropy_summary_file using (strcol(1) eq "Crop" ? $7 : 1/0):(7.8-$5):($13-$7):(0) with vectors nohead linecolor rgb "#E8C2C7" linewidth 1.0 notitle, \
        entropy_summary_file using (strcol(1) eq "Crop" ? $8 : 1/0):(7.8-$5):($12-$8):(0) with vectors nohead linecolor rgb crop_anchored_raw_color linewidth 2.5 notitle, \
        entropy_summary_file using (strcol(1) eq "Crop" ? $9 : 1/0):(7.8-$5):($11-$9):(0) with vectors nohead linecolor rgb crop_anchored_color linewidth 5.0 notitle, \
        entropy_summary_file using (strcol(1) eq "Crop" ? $10 : 1/0):(7.8-$5) with points pointtype 7 pointsize 0.85 linecolor rgb crop_anchored_color notitle, \
        entropy_summary_file using (strcol(1) eq "HalfCheetah" ? $7 : 1/0):(4.2-$5):($13-$7):(0) with vectors nohead linecolor rgb "#C6D3E1" linewidth 1.0 notitle, \
        entropy_summary_file using (strcol(1) eq "HalfCheetah" ? $8 : 1/0):(4.2-$5):($12-$8):(0) with vectors nohead linecolor rgb terminal_raw_color linewidth 2.5 notitle, \
        entropy_summary_file using (strcol(1) eq "HalfCheetah" ? $9 : 1/0):(4.2-$5):($11-$9):(0) with vectors nohead linecolor rgb td3_color linewidth 5.0 notitle, \
        entropy_summary_file using (strcol(1) eq "HalfCheetah" ? $10 : 1/0):(4.2-$5) with points pointtype 7 pointsize 0.85 linecolor rgb td3_color notitle

    unset label 201
    unset label 202
    unset label 203
    unset label 204

    # (c) Seed-level switching on adjacent unique-dominant checkpoints.
    set datafile separator "\t"
    set border 15 front linewidth 1.05 linecolor rgb "#262626"
    unset key
    set grid xtics ytics back linestyle 90
    set lmargin at screen 0.640
    set rmargin at screen 0.975
    set bmargin at screen 0.105
    set tmargin at screen 0.375
    set title "(c) Temporal dominant-head switching" font "Liberation Sans Bold,10.5" offset 0,-0.1
    set xrange [0.55:7.45]
    set yrange [0:0.75]
    set xtics ("Lunar\nNaive" 1, "Lunar\nProtected" 2, "Crop\nNaive" 4, "Crop\nAnchored" 5, "HalfCheetah\nAMAF" 7) font "Liberation Sans,8.2"
    set ytics 0.15
    set mytics 3
    set format y "%.2f"
    unset xlabel
    set ylabel "Dominant-head switch rate" offset 1.0,0
    plot \
        switching_file using (strcol(1) eq "seed" && strcol(2) eq "Lunar" && strcol(3) eq "Naive" ? $6 : 1/0):7 \
            with points pointtype 2 pointsize 0.66 linecolor rgb lunar_naive_color notitle, \
        switching_file using (strcol(1) eq "seed" && strcol(2) eq "Lunar" && strcol(3) eq "Protected" ? $6 : 1/0):7 \
            with points pointtype 8 pointsize 0.62 linecolor rgb lunar_protected_color notitle, \
        switching_file using (strcol(1) eq "seed" && strcol(2) eq "Crop" && strcol(3) eq "Naive" ? $6 : 1/0):7 \
            with points pointtype 2 pointsize 0.66 linecolor rgb crop_naive_color notitle, \
        switching_file using (strcol(1) eq "seed" && strcol(2) eq "Crop" && strcol(3) eq "Anchored" ? $6 : 1/0):7 \
            with points pointtype 13 pointsize 0.62 linecolor rgb crop_anchored_color notitle, \
        switching_file using (strcol(1) eq "seed" && strcol(2) eq "HalfCheetah" ? $6 : 1/0):7 \
            with points pointtype 6 pointsize 0.66 linecolor rgb amaf_color notitle, \
        switching_file using (strcol(1) eq "mean" && strcol(2) eq "Lunar" && strcol(3) eq "Naive" ? $6 : 1/0):7 \
            with points pointtype 2 pointsize 1.35 linecolor rgb lunar_naive_color notitle, \
        switching_file using (strcol(1) eq "mean" && strcol(2) eq "Lunar" && strcol(3) eq "Protected" ? $6 : 1/0):7 \
            with points pointtype 8 pointsize 1.30 linecolor rgb lunar_protected_color notitle, \
        switching_file using (strcol(1) eq "mean" && strcol(2) eq "Crop" && strcol(3) eq "Naive" ? $6 : 1/0):7 \
            with points pointtype 2 pointsize 1.35 linecolor rgb crop_naive_color notitle, \
        switching_file using (strcol(1) eq "mean" && strcol(2) eq "Crop" && strcol(3) eq "Anchored" ? $6 : 1/0):7 \
            with points pointtype 13 pointsize 1.30 linecolor rgb crop_anchored_color notitle, \
        switching_file using (strcol(1) eq "mean" && strcol(2) eq "HalfCheetah" ? $6 : 1/0):7 \
            with points pointtype 6 pointsize 1.35 linecolor rgb amaf_color notitle

    unset multiplot
    unset output
}
