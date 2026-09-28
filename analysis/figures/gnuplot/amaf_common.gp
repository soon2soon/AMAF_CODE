# Shared restrained journal style for AMAF manuscript figures.

set encoding utf8

font_name = "Liberation Sans"
axis_label_font = font_name . ",13"
tic_label_font = font_name . ",11"
legend_font = font_name . ",10.5"

td3_color = "#355F8A"
amaf_color = "#B23A48"
not_improved_color = "#4C6A92"
terminal_raw_color = "#7895B0"
late_gain_raw_color = "#C97A84"
lunar_uniform_color = "#355F8A"
lunar_naive_color = "#B23A48"
lunar_protected_color = "#3F7F6C"
lunar_dqn_color = "#545454"
lunar_dueling_color = "#7A7A7A"
crop_anchored_color = "#B23A48"
crop_uniform_color = "#355F8A"
crop_naive_color = "#A85472"
crop_dueling_color = "#3F7F68"
crop_dqn_color = "#666666"
crop_anchored_raw_color = "#C8707B"
crop_uniform_raw_color = "#6889A8"
crop_naive_raw_color = "#C17A91"
crop_dueling_raw_color = "#6E9C8D"
crop_dqn_raw_color = "#919191"
benefit_positive_color = "#3567A8"
benefit_negative_color = "#C43C4A"

set border 15 front linewidth 1.05 linecolor rgb "#262626"
set tics in mirror scale 0.60,0.32 font tic_label_font
set xtics mirror
set ytics mirror
unset x2tics
unset y2tics
set mxtics 2
set mytics 2

set style line 90 linecolor rgb "#D4D4D4" linewidth 0.55 dashtype 2
set style line 91 linecolor rgb "#ECECEC" linewidth 0.35 dashtype 2
set grid xtics ytics mxtics mytics back linestyle 90, linestyle 91

set xlabel font axis_label_font
set ylabel font axis_label_font
set key opaque box linewidth 0.7 samplen 2.5 spacing 1.0 font legend_font

# Both means use solid lines; sparse markers preserve grayscale distinction.
set style line 1 linecolor rgb td3_color linewidth 2.0 dashtype 1
set style line 2 linecolor rgb amaf_color linewidth 2.0 dashtype 1
set style line 11 linecolor rgb td3_color linewidth 2.0 dashtype 1 pointtype 4 pointsize 0.80
set style line 12 linecolor rgb amaf_color linewidth 2.0 dashtype 1 pointtype 2 pointsize 0.95
set style line 20 linecolor rgb "#3F3F3F" linewidth 0.85 dashtype 1
set style line 21 linecolor rgb "#888888" linewidth 0.75 dashtype 2
set style line 22 linecolor rgb "#A8A8A8" linewidth 0.65 dashtype 1

# Semantic colors retain circle/triangle cohort redundancy.
set style line 31 linecolor rgb amaf_color linewidth 1.65 pointtype 6 pointsize 1.30
set style line 32 linecolor rgb amaf_color linewidth 1.65 pointtype 8 pointsize 1.35
set style line 33 linecolor rgb not_improved_color linewidth 1.05 pointtype 6 pointsize 1.30
set style line 34 linecolor rgb not_improved_color linewidth 1.05 pointtype 8 pointsize 1.35

set style line 41 linecolor rgb terminal_raw_color linewidth 1.05 pointtype 6 pointsize 1.30
set style line 42 linecolor rgb terminal_raw_color linewidth 1.05 pointtype 8 pointsize 1.35
set style line 43 linecolor rgb late_gain_raw_color linewidth 1.05 pointtype 6 pointsize 1.30
set style line 44 linecolor rgb late_gain_raw_color linewidth 1.05 pointtype 8 pointsize 1.35
set style line 45 linecolor rgb td3_color linewidth 1.55 pointtype 13 pointsize 1.45
set style line 46 linecolor rgb amaf_color linewidth 1.55 pointtype 13 pointsize 1.45
set style line 47 linecolor rgb terminal_raw_color linewidth 1.65 pointtype 6 pointsize 1.30
set style line 48 linecolor rgb terminal_raw_color linewidth 1.65 pointtype 8 pointsize 1.35
set style line 49 linecolor rgb late_gain_raw_color linewidth 1.65 pointtype 6 pointsize 1.30
set style line 50 linecolor rgb late_gain_raw_color linewidth 1.65 pointtype 8 pointsize 1.35

# Lunar learning curves use solid lines; color, marker, and weight encode method.
set style line 61 linecolor rgb lunar_uniform_color linewidth 2.0 dashtype 1 pointtype 4 pointsize 0.80
set style line 62 linecolor rgb lunar_naive_color linewidth 2.0 dashtype 1 pointtype 2 pointsize 0.95
set style line 63 linecolor rgb lunar_protected_color linewidth 2.0 dashtype 1 pointtype 8 pointsize 0.90
set style line 64 linecolor rgb lunar_dqn_color linewidth 1.25 dashtype 2 pointtype 6 pointsize 0.62
set style line 65 linecolor rgb lunar_dueling_color linewidth 1.15 dashtype 3 pointtype 8 pointsize 0.60

# Crop: method identity is shared by performance and resource-yield panels.
set style line 71 linecolor rgb crop_dqn_color linewidth 1.10 dashtype 1 pointtype 6 pointsize 0.72
set style line 72 linecolor rgb crop_dueling_color linewidth 1.25 dashtype 1 pointtype 8 pointsize 0.76
set style line 73 linecolor rgb crop_uniform_color linewidth 1.55 dashtype 1 pointtype 4 pointsize 0.82
set style line 74 linecolor rgb crop_naive_color linewidth 1.35 dashtype 1 pointtype 2 pointsize 0.88
set style line 75 linecolor rgb crop_anchored_color linewidth 1.65 dashtype 1 pointtype 13 pointsize 0.90
