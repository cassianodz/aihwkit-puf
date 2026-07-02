"""
puf/metrics.py — PUF quality metrics (Maiti & Schaumont 2011, IEEE Trans. VLSI)

Uniformidade é medida no nível de POPULAÇÃO (Eq. 3 do paper),
não por dispositivo individual.
"""
import numpy as np
from itertools import combinations

class PUFMetrics:

    @staticmethod
    def uniqueness(signatures):
        if len(signatures) < 2:
            raise ValueError("Precisa de ao menos 2 assinaturas.")
        d = np.array([float(np.mean(a.astype(int) != b.astype(int)))
                      for a, b in combinations(signatures, 2)])
        return {"metric": "uniqueness", "ideal": 0.50,
                "mean": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
                "n_pairs": len(d), "passed": 0.45 <= float(d.mean()) <= 0.55}

    @staticmethod
    def reliability(sigs_over_time, reference_idx=0):
        if len(sigs_over_time) < 2:
            raise ValueError("Precisa de ao menos 2 pontos no tempo.")
        ref  = sigs_over_time[reference_idx]
        sims = [float(np.mean(ref.astype(int) == s.astype(int)))
                for i, s in enumerate(sigs_over_time) if i != reference_idx]
        mean_sim = float(np.mean(sims))
        return {"metric": "reliability", "ideal": ">0.90",
                "mean_similarity": round(mean_sim, 4),
                "min_similarity":  round(float(np.min(sims)), 4),
                "passed": mean_sim >= 0.90}

    @staticmethod
    def uniformity_population(enrollment_signatures):
        all_bits   = np.concatenate([s.astype(int) for s in enrollment_signatures])
        f          = float(all_bits.mean())
        per_device = [float(s.mean()) for s in enrollment_signatures]
        n_out      = sum(1 for p in per_device if not (0.45 <= p <= 0.55))
        return {"metric": "uniformity_population", "ideal": 0.50,
                "population_mean": round(f, 4),
                "per_device_std": round(float(np.std(per_device)), 4),
                "n_devices_outside_45_55": n_out,
                "passed": 0.45 <= f <= 0.55}

    @staticmethod
    def full_report(signatures_per_device, verbose=True):
        if len(signatures_per_device) < 2:
            raise ValueError("Precisa de ao menos 2 dispositivos.")
        enrollment = [d[0] for d in signatures_per_device]
        u    = PUFMetrics.uniqueness(enrollment)
        rels = [PUFMetrics.reliability(d) for d in signatures_per_device if len(d) >= 2]
        mean_rel = float(np.mean([r["mean_similarity"] for r in rels]))
        min_rel  = float(np.min([r["min_similarity"]  for r in rels]))
        rel_ok   = all(r["passed"] for r in rels)
        unif     = PUFMetrics.uniformity_population(enrollment)
        overall  = u["passed"] and rel_ok and unif["passed"]
        if verbose:
            sep = "=" * 60
            print(sep); print("PUF QUALITY REPORT"); print(sep)
            print(f"Devices evaluated : {len(signatures_per_device)}")
            print(f"Bits per signature: {len(enrollment[0])}")
            print(sep)
            print(f"Uniqueness   : {u['mean']:.3f} \u00b1 {u['std']:.3f}  (ideal ~0.50)  {'\u2713' if u['passed'] else '\u2717'}")
            print(f"Reliability  : {mean_rel:.3f} (min {min_rel:.3f})  (ideal >0.90)  {'\u2713' if rel_ok else '\u2717'}")
            print(f"Uniformity   : {unif['population_mean']:.3f}  [per-device std={unif['per_device_std']:.3f}]  (ideal ~0.50)  {'\u2713' if unif['passed'] else '\u2717'}")
            if unif["n_devices_outside_45_55"] > 0:
                print(f"             \u2192 {unif['n_devices_outside_45_55']} disp. com uniformidade individual fora de [0.45,0.55]")
                print(f"             \u2192 Normal para PUFs com offset f\u00edsico (ver Maiti 2011)")
            print("-" * 60)
            print(f"Overall      : {'PASSED \u2713' if overall else 'FAILED \u2717'}")
            print(sep)
        return {"uniqueness": u, "reliability": {"mean": mean_rel, "min": min_rel, "passed": rel_ok},
                "uniformity": unif, "overall_passed": overall}
