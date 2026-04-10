
import io
import numpy as np
import lasio
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def clean_curve(depth, curve, null_value):
    mask = np.isfinite(depth) & np.isfinite(curve) & (curve != null_value)
    return depth[mask], curve[mask]


def normalise(x):
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def read_uploaded_las(uploaded_file):
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore")
    return lasio.read(io.StringIO(text))


def get_curve_names(las):
    return [curve.mnemonic for curve in las.curves]


def depth_match_from_single_las(
    las,
    ref_curve,
    run_curve,
    depth_curve="DEPT",
    resample_step=0.01,
    shift_min=-5.5,
    shift_max=5.5,
    shift_step=0.001,
    smooth_sigma=3.0,
):
    if depth_curve in las.curves:
        depth = np.array(las[depth_curve], dtype=float)
    else:
        depth = np.array(las.index, dtype=float)

    if ref_curve not in las.curves:
        raise ValueError(f"Curve '{ref_curve}' not found in LAS file.")

    if run_curve not in las.curves:
        raise ValueError(f"Curve '{run_curve}' not found in LAS file.")

    gamma_ref = np.array(las[ref_curve], dtype=float)
    gamma_run = np.array(las[run_curve], dtype=float)

    null_value = float(las.well.NULL.value) if "NULL" in las.well else -999.25

    depth_ref, gamma_ref = clean_curve(depth, gamma_ref, null_value)
    depth_run, gamma_run = clean_curve(depth, gamma_run, null_value)

    zmin = max(np.min(depth_ref), np.min(depth_run))
    zmax = min(np.max(depth_ref), np.max(depth_run))

    if zmax <= zmin:
        raise ValueError("No overlapping depth interval found.")

    common_depth = np.arange(zmin, zmax, resample_step)

    f_ref = interp1d(depth_ref, gamma_ref, bounds_error=False, fill_value=np.nan)
    f_run = interp1d(depth_run, gamma_run, bounds_error=False, fill_value=np.nan)

    ref = f_ref(common_depth)
    run = f_run(common_depth)

    ref = gaussian_filter1d(ref, sigma=smooth_sigma, mode="nearest")
    run = gaussian_filter1d(run, sigma=smooth_sigma, mode="nearest")

    shifts = np.arange(shift_min, shift_max + shift_step, shift_step)
    correlations = []

    for shift in shifts:
        shifted_run = interp1d(
            common_depth,
            run,
            bounds_error=False,
            fill_value=np.nan
        )(common_depth + shift)

        valid = np.isfinite(ref) & np.isfinite(shifted_run)

        if np.sum(valid) < 20:
            correlations.append(np.nan)
            continue

        x = normalise(ref[valid])
        y = normalise(shifted_run[valid])
        corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    best_idx = np.nanargmax(correlations)
    best_shift_internal = shifts[best_idx]
    best_corr = correlations[best_idx]

    # Display convention retained from your notebook
    best_shift_display = -best_shift_internal

    best_shifted_run = interp1d(
        common_depth,
        run,
        bounds_error=False,
        fill_value=np.nan
    )(common_depth + best_shift_internal)

    shifted_depth_axis = common_depth + best_shift_display
    display_shifts = -shifts

    return {
        "best_shift_internal": best_shift_internal,
        "best_shift_display": best_shift_display,
        "best_corr": best_corr,
        "common_depth": common_depth,
        "ref": ref,
        "run": run,
        "shifted_run": best_shifted_run,
        "shifted_depth_axis": shifted_depth_axis,
        "shifts": shifts,
        "display_shifts": display_shifts,
        "correlations": correlations,
    }


def make_before_plot(result, ref_curve, run_curve):
    fig, ax = plt.subplots(figsize=(4.5, 6))

    ax.plot(
        result["ref"],
        result["common_depth"],
        linewidth=1.5,
        label=f"Reference, {ref_curve}"
    )

    ax.plot(
        result["run"],
        result["common_depth"],
        linewidth=1.5,
        alpha=0.8,
        label=f"Curve to Shift, {run_curve}"
    )

    ax.invert_yaxis()
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Before Matching")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig

def make_after_plot(result, ref_curve, run_curve):
    fig, ax = plt.subplots(figsize=(4.5, 6))

    ax.plot(
        result["ref"],
        result["common_depth"],
        linewidth=1.5,
        label=f"Reference, {ref_curve}"
    )

    ax.plot(
        result["shifted_run"],
        result["common_depth"],
        linewidth=1.5,
        linestyle="--",
        label=f"Shifted, {run_curve}"
    )

    ax.invert_yaxis()
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Depth (m)")
    ax.set_title("After Matching")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def make_correlation_plot(result):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(result["display_shifts"], result["correlations"])
    ax.axvline(
        result["best_shift_display"],
        linestyle="--",
        label=f"Best shift = {result['best_shift_display']:.3f} m",
    )
    ax.set_xlabel("Shift (m)")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation vs Shift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Depth Matching", layout="centered")
st.title("Depth Matching for Gamma Curves")

uploaded_file = st.file_uploader("Upload a LAS file", type=["las"])

if uploaded_file is not None:
    try:
        las = read_uploaded_las(uploaded_file)
        curve_names = [curve.mnemonic for curve in las.curves]

        curve_options = [
            c for c in curve_names
            if any(k in c.upper() for k in ["NG", "GAMMA", "GR"])
        ]

        if not curve_options:
            curve_options = curve_names

        if not curve_names:
            st.error("No curves were found in the uploaded LAS file.")
            st.stop()

        st.subheader("Curve selection")

        col1, col2 = st.columns(2)

        with col1:
            ref_curve = st.selectbox(
                "Reference curve",
                options=curve_options,
                index=0
            )

        with col2:
            run_default_index = 1 if len(curve_options) > 1 else 0
            run_curve = st.selectbox(
                "Curve to Shift",
                options=curve_options,
                index=run_default_index
            )

        st.subheader("Matching parameters")

        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            resample_step = st.number_input("resample_step", value=0.01, format="%.3f")
        with p2:
            shift_min = st.number_input("shift_min", value=-5.5, format="%.3f")
        with p3:
            shift_max = st.number_input("shift_max", value=5.5, format="%.3f")
        with p4:
            shift_step = st.number_input("shift_step", value=0.001, format="%.4f")
        with p5:
            smooth_sigma = st.number_input(
                "smooth_sigma",
                 min_value=0.0,
                 max_value=20.0,
                 value=3.0,
                 step=0.1,
                 format="%.1f"
                 )

        if st.button("Run depth matching", type="primary"):
            if ref_curve == run_curve:
                st.warning("Please select two different curves.")
                st.stop()
            if shift_step <= 0:
                st.error("shift_step must be greater than 0.")
                st.stop()
            if resample_step <= 0:
                st.error("resample_step must be greater than 0.")
                st.stop()
            if shift_max <= shift_min:
                st.error("shift_max must be greater than shift_min.")
                st.stop()

            result = depth_match_from_single_las(
                las=las,
                ref_curve=ref_curve,
                run_curve=run_curve,
                depth_curve="DEPT",
                resample_step=resample_step,
                shift_min=shift_min,
                shift_max=shift_max,
                shift_step=shift_step,
                smooth_sigma=smooth_sigma,
            )

            st.subheader("Results")

            c1, c2 = st.columns(2)
            with c1:
                if result["best_shift_display"] < 0:
                    st.metric("Recommended shift", f"{result['best_shift_display']:.3f} m UP")
                elif result["best_shift_display"] > 0:
                    st.metric("Recommended shift", f"{result['best_shift_display']:.3f} m DOWN")
                else:
                    st.metric("Recommended shift", "0.000 m")
            with c2:
                st.metric("Best correlation", f"{result['best_corr']:.3f}")

            st.write(f"Reference curve: **{ref_curve}**")
            st.write(f"Curve to Shift: **{run_curve}**")

            fig_before = make_before_plot(result, ref_curve, run_curve)
            fig_after = make_after_plot(result, ref_curve, run_curve)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_before)
            with col2:
                st.pyplot(fig_after)
            
            fig_corr = make_correlation_plot(result)
            st.pyplot(fig_corr)

            corr_df = pd.DataFrame({
                "display_shift_m": result["display_shifts"],
                "correlation": result["correlations"],
            })
            st.download_button(
                "Download correlation table as CSV",
                corr_df.to_csv(index=False).encode("utf-8"),
                file_name="correlation_vs_shift.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a LAS file to begin.")
