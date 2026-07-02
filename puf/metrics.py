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
            u_ok   = "OK" if u["passed"]    else "FAIL"
            r_ok   = "OK" if rel_ok         else "FAIL"
            un_ok  = "OK" if unif["passed"] else "FAIL"
            ov_ok  = "PASSED" if overall    else "FAILED"
            sep = "=" * 60
            print(sep)
            print("PUF QUALITY REPORT")
            print(sep)
            print("Devices evaluated : " + str(len(signatures_per_device)))
            print("Bits per signature: " + str(len(enrollment[0])))
            print(sep)
            print("Uniqueness   : " + str(u["mean"]) + " +/- " + str(u["std"]) +
                  "  (ideal ~0.50)  [" + u_ok + "]")
            print("Reliability  : " + str(mean_rel) + " (min " + str(min_rel) + ")" +
                  "  (ideal >0.90)  [" + r_ok + "]")
            print("Uniformity   : " + str(unif["population_mean"]) +
                  "  [per-device std=" + str(unif["per_device_std"]) + "]" +
                  "  (ideal ~0.50)  [" + un_ok + "]")
            if unif["n_devices_outside_45_55"] > 0:
                print("             -> " + str(unif["n_devices_outside_45_55"]) +
                      " disp. com uniformidade individual fora de [0.45, 0.55]")
                print("             -> Normal para PUFs com offset fisico (Maiti 2011)")
            print("-" * 60)
            print("Overall      : " + ov_ok)
            print(sep)

        return {"uniqueness": u,
                "reliability": {"mean": mean_rel, "min": min_rel, "passed": rel_ok},
                "uniformity": unif, "overall_passed": overall}
