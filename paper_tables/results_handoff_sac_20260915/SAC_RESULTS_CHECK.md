# HalfCheetah SAC primary-five Results check

## Scope

Training과 re-run 없이 frozen/canonical evidence만 사용한 manuscript-safe handoff이다.
Experimental unit은 seed이며 seed 집합은 11, 22, 33, 44, 55이다. 세 방법 모두 10k
간격 evaluation checkpoint를 1.5M까지 동일하게 사용한다.

## Sources

- SAC registry ID: `halfcheetah_sac_primary5`
- SAC path: `paper_data/halfcheetah/sac_primary5_1p5m_20260910`
- lifecycle/manuscript allowed: `frozen` / `true`
- SAC SHA256SUMS_sha256: `0c5795b36a614822c17bb79a26e2db41318c5ba1d41cc8ca7f428fc3848f8f7c`
- TD3/AMAF canonical registry ID: `halfcheetah_per_seed`
- TD3/AMAF canonical fingerprint: `7856c4cd2a32bd28e8a067ddd8025fe031799b1a464a305e6d4a6ea6cc554082`

SAC source path는 `figure_contract.yaml` registry로 해결했다. MANIFEST에 선언된 canonical
파일과 검증된 SHA-256은 다음과 같다.

- `paper_tables/canonical/halfcheetah_sac_per_seed.csv`: `35d27a1692b643e4b9eaa44f4259a295e0412b5da222bf4ac37ced2047e6f739`
- `paper_tables/canonical/halfcheetah_sac_aggregate.csv`: `4321cafad5b2103d3c37af41b5ae066216b3e5d6be52ffa3f3c424291bc34f12`
- `paper_tables/canonical/halfcheetah_primary5_td3_amaf_sac_per_seed.csv`: `61606ef738f495a18b5bff4c47761ac1d8421d4df7eabe348f20d186cab4c7a5`
- `paper_tables/canonical/halfcheetah_primary5_td3_amaf_sac_aggregate.csv`: `00f2ed12d16fe52aa49c8c52ea6f2958bf636535d4bf270a8d7863ca3c7e8412`
- `paper_tables/canonical/halfcheetah_primary5_td3_amaf_sac_pairwise.csv`: `8f6e2664cfd2158f838deaec867f0d56dd1d7da568cec413213dd239e8a4fd8a`

## Metrics and bootstrap

- 1M/1.5M: 정확한 checkpoint의 evaluation-episode mean return
- gain: 1.5M return − 1M return
- Last-5/Last-10: 마지막 5/10 checkpoint mean return의 평균
- normalized AUC: 10k–1.5M checkpoint mean curve의 시간 정규화 trapezoidal AUC
- paired delta: 모두 AMAF − SAC. gain delta는 두 방법 gain의 차이
- bootstrap: seed-level percentile mean CI, 100,000 resamples,
  95% confidence, RNG seed 20260907

## Primary-five method means

- TD3: 1M 10622.36; 1.5M 11711.40; gain 1089.04; Last-5 11676.75; Last-10 11641.36; AUC 9232.05
- AMAF: 1M 10275.82; 1.5M 12046.86; gain 1771.05; Last-5 11926.21; Last-10 11799.37; AUC 9108.73
- SAC: 1M 10859.52; 1.5M 11780.38; gain 920.87; Last-5 11862.65; Last-10 11822.04; AUC 9536.05

## AMAF − SAC paired effects

- terminal_delta: mean +266.48; median -234.64; 95% CI [-906.12, +1596.81]; wins/losses/ties 2/3/0
- gain_delta: mean +850.18; median +814.71; 95% CI [+482.40, +1252.38]; wins/losses/ties 5/0/0
- last5_delta: mean +63.56; median -226.94; 95% CI [-1126.23, +1317.91]; wins/losses/ties 2/3/0
- last10_delta: mean -22.67; median -204.83; 95% CI [-1226.35, +1306.07]; wins/losses/ties 2/3/0
- auc_delta: mean -427.32; median -688.35; 95% CI [-1566.87, +712.23]; wins/losses/ties 2/3/0

## Verification

- Frozen checkpoint data에서 세 방법의 모든 필수 metric을 재계산했다.
- Registered TD3/AMAF canonical, standalone SAC canonical, combined three-method canonical,
  aggregate canonical, AMAF-SAC paired canonical과 수치가 일치했다.
- 요청에 제공된 method mean 및 AMAF-SAC mean/median/wins sanity fingerprint와 모두 일치했다.
- 출력 행 수, seed 집합, duplicate key, NaN, wins/losses/ties 합계를 검증했다.
- 수치 계산에 사용한 `MANIFEST.json`, 5개 `COMPLETED`, 5개 `eval_metrics.csv`는
  `SHA256SUMS`와 모두 일치했다.

전체 `sha256sum -c SHA256SUMS`는 목록에 선언됐지만 frozen tree에 없는 model-weight
파일 15개 때문에 PASS가 아니다. 누락 항목은 각 seed의 `checkpoints/best.pt`,
`checkpoints/final.pt`, `checkpoints/latest.pt`이며 numerical table 계산에는 사용되지
않았다. 따라서 Results 수치는 frozen evaluation input으로 재현 및 검증되지만, 이
디렉터리를 checkpoint weight까지 완전한 archive로 기술하면 안 된다.

## Results-safe reading

SAC는 primary-five 평균에서 1M return과 normalized AUC가 AMAF보다 높았다. AMAF는
1M→1.5M gain이 SAC보다 컸으며 paired gain delta는 5/5 seed에서 양수이고 95% CI도
0보다 높았다. AMAF terminal mean은 SAC보다 높았지만 terminal wins는 2/5이고 CI가
0을 포함하므로 seed-wise dominance로 기술하면 안 된다.
