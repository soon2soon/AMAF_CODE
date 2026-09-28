# HalfCheetah Panel C: paired effect summary.
# Run from the repository root:
#   gnuplot analysis/figures/gnuplot/halfcheetah/panel_c_effect_summary.gp

load "analysis/figures/gnuplot/amaf_common.gp"

raw_data = "analysis/figures/figure_data/halfcheetah_panel_b_transition.csv"
summary_data = "analysis/figures/figure_data/halfcheetah_panel_c_effect_summary.csv"
pdf_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_c_effect_summary.pdf"
png_output = "paper_figures/gnuplot/halfcheetah/halfcheetah_panel_c_effect_summary.png"

set datafile separator comma
set xrange [-4200:2500]
set yrange [0.55:2.50]
set xtics 1000
set ytics ("Late-stage gain" 1, "Terminal (1.5M)" 2)
unset mytics
set format x "%.0f"
set xlabel "Paired effect (AMAF - TD3)"
unset ylabel
set key top center horizontal at graph 0.50,0.975
set bars 0.8

set lmargin at screen 0.285
set rmargin at screen 0.985
set bmargin at screen 0.180
set tmargin at screen 0.970

set arrow 1 from first 0, graph 0 to first 0, graph 1 nohead back linestyle 20

set terminal pdfcairo enhanced color font "Liberation Sans,10.5" size 3.70in,2.25in
set output pdf_output
plot \
    raw_data using (strcol(2) eq "original5" && $5 <= 0 ? $5 : 1/0):(1.08) \
        with points linestyle 43 notitle, \
    raw_data using (strcol(2) eq "replication5" && $5 <= 0 ? $5 : 1/0):(0.92) \
        with points linestyle 44 notitle, \
    raw_data using (strcol(2) eq "original5" && $4 <= 0 ? $4 : 1/0):(2.08) \
        with points linestyle 41 notitle, \
    raw_data using (strcol(2) eq "replication5" && $4 <= 0 ? $4 : 1/0):(1.92) \
        with points linestyle 42 notitle, \
    raw_data using (strcol(2) eq "original5" && $5 > 0 ? $5 : 1/0):(1.08) \
        with points linestyle 49 notitle, \
    raw_data using (strcol(2) eq "replication5" && $5 > 0 ? $5 : 1/0):(0.92) \
        with points linestyle 50 notitle, \
    raw_data using (strcol(2) eq "original5" && $4 > 0 ? $4 : 1/0):(2.08) \
        with points linestyle 47 title "Original 5", \
    raw_data using (strcol(2) eq "replication5" && $4 > 0 ? $4 : 1/0):(1.92) \
        with points linestyle 48 title "Replication 5", \
    summary_data using ($2 == 2 ? $4 : 1/0):2:6:7 \
        with xerrorbars linestyle 45 title "Mean, 95% CI", \
    summary_data using ($2 == 1 ? $4 : 1/0):2:6:7 \
        with xerrorbars linestyle 46 notitle, \
    summary_data using 11:2:10 with labels right font legend_font \
        textcolor rgb "#333333" offset -0.4,0 notitle
unset output

set terminal pngcairo enhanced color font "Liberation Sans,10.5" size 1110,675
set output png_output
replot
unset output
