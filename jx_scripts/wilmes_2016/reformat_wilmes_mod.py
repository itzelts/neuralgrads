#!/usr/bin/env python3
"""
reformat_wilmes_mod.py

Reformats Wilmes_2016 MOD files to be compatible with DendroTweaks' MODFileConverter.
Applies Hay_2011 formatting conventions channel-by-channel.

Usage:
    python jx_scripts/reformat_wilmes_mod.py \
        --input-dir examples/Wilmes_2016/biophys/mod \
        --output-dir examples/Wilmes_2016/biophys/tempmod
"""

import os
import re
import argparse


# ---------------------------------------------------------------------------
# Group A: na3 family — sodium channels with 3 states (m, h, s)
# Root cause: PROCEDURE trates(vm, a2) has two params; converter only
# supports one param (the independent variable).
# Fix: rename to rates(v(mV)), inline a2→ar2 from PARAMETER.
# ---------------------------------------------------------------------------

def transform_na3(content: str) -> str:
    """Fix na3.mod: two-param PROCEDURE trates(vm,a2) → rates(v(mV))."""
    # Remove GLOBAL line from NEURON block (grammar suppresses it but noisy)
    content = re.sub(r'\bGLOBAL\b[^\n]*\n', '', content)

    # Remove standalone LOCAL outside any block
    content = re.sub(r'^LOCAL\s+\w[\w,\s]*\n', '', content, flags=re.MULTILINE)

    # Rename PROCEDURE signature: trates(vm,a2) → rates(v(mV))
    content = re.sub(
        r'\bPROCEDURE\s+trates\s*\(\s*vm\s*,\s*a2\s*\)',
        'PROCEDURE rates(v(mV))',
        content
    )

    # Rename parameter vm→v in the procedure body (word boundary safe)
    content = re.sub(r'\bvm\b', 'v', content)

    # Replace a2 with ar2 in the body (ar2 remains in PARAMETER)
    content = re.sub(r'\ba2\b', 'ar2', content)

    # Fix call site: trates(v+vshift, ar2) → rates(v+vshift)  [na3dend/shifted]
    content = re.sub(
        r'\btrates\s*\(\s*v\s*\+\s*vshift\s*,\s*ar2\s*\)',
        'rates(v+vshift)',
        content
    )

    # Fix call site: trates(v, ar2) → rates(v)  [na3]
    content = re.sub(
        r'\btrates\s*\(\s*v\s*,\s*ar2\s*\)',
        'rates(v)',
        content
    )

    return content


# ---------------------------------------------------------------------------
# Group B: kdrca1 / kap — delayed-rectifier and A-type K channels
# These likely work as-is, but non-standard conductance variable names
# (gkdrbar, gkdr, gkabar, gka) are renamed to gbar/g for Hay consistency.
# Apply longer name first to avoid partial-match corruption.
# ---------------------------------------------------------------------------

def transform_kdrca1(content: str) -> str:
    """Rename conductance vars and clean up kdrca1.mod."""
    content = re.sub(r'\bGLOBAL\b[^\n]*\n', '', content)
    # Rename longer name first (gkdrbar → gbar) before shorter (gkdr → g)
    content = re.sub(r'\bgkdrbar\b', 'gbar', content)
    content = re.sub(r'\bgkdr\b', 'g', content)
    # Update RANGE statement to Hay style
    content = re.sub(
        r'\bRANGE\s+g\s*,\s*gbar\b',
        'RANGE gbar, g, ik',
        content
    )
    # Fallback: if RANGE still looks like original
    content = re.sub(
        r'\bRANGE\s+gbar\s*,\s*g\b(?!\s*,\s*ik)',
        'RANGE gbar, g, ik',
        content
    )
    return content


def transform_kap(content: str) -> str:
    """Rename conductance vars and clean up kap.mod."""
    content = re.sub(r'\bGLOBAL\b[^\n]*\n', '', content)
    # Rename longer name first
    content = re.sub(r'\bgkabar\b', 'gbar', content)
    content = re.sub(r'\bgka\b', 'g', content)
    # Update RANGE
    content = re.sub(
        r'\bRANGE\s+gbar\s*,\s*g\b(?!\s*,\s*ik)',
        'RANGE gbar, g, ik',
        content
    )
    return content


# ---------------------------------------------------------------------------
# Group C: sca / kca / it2 — channels using TABLE-based forward Euler
# These need full structural rewrites.
# Root causes: SOLVE without METHOD, TABLE/DEPEND in PROCEDURE, two
# PROCEDURE blocks, no DERIVATIVE block.
# Strategy: generate fresh MOD strings from the original kinetics.
# ---------------------------------------------------------------------------

# sca.mod — HVA calcium channel (same biophysical basis as CaHVA2.mod)
# Original kinetics from rates(vm): alpha-beta formulas with actF / inactF
# scaling.  Original BREAKPOINT used tadj and a (1e-4) unit factor; both are
# dropped here (gbar moved to S/cm2, tadj handled by Jaxley framework).
SCA_MOD = """\

NEURON {
\tSUFFIX sca
\tUSEION ca READ eca WRITE ica
\tRANGE gbar, g, ica
}

UNITS {
\t(S) = (siemens)
\t(mV) = (millivolt)
\t(mA) = (milliamp)
}

PARAMETER {
\tgbar = 0.0 (S/cm2)
\tvshift = 0 (mV)
\tactF = 1
\tinactF = 3
\tq10 = 2.3
\ttemp = 23 (degC)
}

ASSIGNED {
\tv\t(mV)
\teca\t(mV)
\tica\t(mA/cm2)
\tg\t(S/cm2)
\tcelsius (degC)
\tmInf
\tmTau\t(ms)
\thInf
\thTau\t(ms)
}

STATE {
\tm
\th
}

BREAKPOINT {
\tSOLVE states METHOD cnexp
\tg = gbar*m*m*h
\tica = g*(v-eca)
}

DERIVATIVE states {
\trates(v+vshift)
\tm' = (mInf-m)/mTau
\th' = (hInf-h)/hTau
}

INITIAL {
\trates(v+vshift)
\tm = mInf
\th = hInf
}

PROCEDURE rates(v(mV)) {
\tLOCAL a, b, qt
\tqt = q10^((celsius-temp)/10)

\ta = 0.055*(-27-v)/(exp((-27-v)/3.8) - 1)/actF
\tb = 0.94*exp((-75-v)/17)/actF
\tmTau = 1/(a+b)/qt
\tmInf = a/(a+b)

\ta = 0.000457*exp((-13-v)/50)/inactF
\tb = 0.0065/(exp((-v-15)/28) + 1)/inactF
\thTau = 1/(a+b)/qt
\thInf = a/(a+b)
}
"""


# kca.mod — calcium-activated K channel (analogous to SK_E2.mod)
# Original kinetics: a = Ra*cai^caix, b = Rb.
# Independent variable is cai, not v.
# Original BREAKPOINT used tadj and (1e-4) unit factor; both dropped here.
KCA_MOD = """\

NEURON {
\tSUFFIX kca
\tUSEION k READ ek WRITE ik
\tUSEION ca READ cai
\tRANGE gbar, g, ik
}

UNITS {
\t(S) = (siemens)
\t(mV) = (millivolt)
\t(mA) = (milliamp)
\t(mM) = (milli/liter)
}

PARAMETER {
\tgbar = 0.0 (S/cm2)
\tcaix = 1
\tRa = 0.01 (/ms)
\tRb = 0.02 (/ms)
\tq10 = 2.3
\ttemp = 23 (degC)
}

ASSIGNED {
\tv\t(mV)
\tek\t(mV)
\tcai\t(mM)
\tik\t(mA/cm2)
\tg\t(S/cm2)
\tcelsius (degC)
\tnInf
\tnTau\t(ms)
}

STATE {
\tn
}

BREAKPOINT {
\tSOLVE states METHOD cnexp
\tg = gbar*n
\tik = g*(v-ek)
}

DERIVATIVE states {
\trates(cai)
\tn' = (nInf-n)/nTau
}

INITIAL {
\trates(cai)
\tn = nInf
}

PROCEDURE rates(cai(mM)) {
\tLOCAL a, qt
\tqt = q10^((celsius-temp)/10)

\ta = Ra*cai^caix
\tnTau = 1/(a+Rb)/qt
\tnInf = a/(a+Rb)
}
"""


# it2.mod — T-type (LVA) calcium channel (analogous to CaLVAst.mod)
# Kinetics: Boltzmann steady-state and empirical tau expressions, with
# vshift voltage offset.  No temperature compensation in original BREAKPOINT.
# Renamed: gcabar→gbar, gca→g.
IT2_MOD = """\

NEURON {
\tSUFFIX it2
\tUSEION ca READ eca WRITE ica
\tRANGE gbar, g, ica
}

UNITS {
\t(S) = (siemens)
\t(mV) = (millivolt)
\t(mA) = (milliamp)
}

PARAMETER {
\tgbar = 0.0 (S/cm2)
\tvshift = 0 (mV)
\tv12m = 50 (mV)
\tv12h = 78 (mV)
\tvwm = 7.4 (mV)
\tvwh = 5.0 (mV)
\tam = 3 (mV)
\tah = 85 (mV)
\tvm1 = 25 (mV)
\tvm2 = 100 (mV)
\tvh1 = 46 (mV)
\tvh2 = 405 (mV)
\twm1 = 20 (mV)
\twm2 = 15 (mV)
\twh1 = 4 (mV)
\twh2 = 50 (mV)
}

ASSIGNED {
\tv\t(mV)
\teca\t(mV)
\tica\t(mA/cm2)
\tg\t(S/cm2)
\tmInf
\tmTau\t(ms)
\thInf
\thTau\t(ms)
}

STATE {
\tm
\th
}

BREAKPOINT {
\tSOLVE states METHOD cnexp
\tg = gbar*m*m*h
\tica = g*(v-eca)
}

DERIVATIVE states {
\trates(v+vshift)
\tm' = (mInf-m)/mTau
\th' = (hInf-h)/hTau
}

INITIAL {
\trates(v+vshift)
\tm = mInf
\th = hInf
}

PROCEDURE rates(v(mV)) {
\tmInf = 1.0/(1 + exp(-(v+v12m)/vwm))
\thInf = 1.0/(1 + exp((v+v12h)/vwh))
\tmTau = am + 1.0/(exp((v+vm1)/wm1) + exp(-(v+vm2)/wm2))
\thTau = ah + 1.0/(exp((v+vh1)/wh1) + exp(-(v+vh2)/wh2))
}
"""


# ---------------------------------------------------------------------------
# Dispatch table and main logic
# ---------------------------------------------------------------------------

SKIP_FILES = {
    'hh2.mod': 'multi-ion channel (Na+K) — multiple USEION WRITE not supported',
    'hh3.mod': 'multi-ion channel (Na+K) — multiple USEION WRITE not supported',
    'cad2.mod': 'calcium dynamics mechanism — no PROCEDURE block, cannot convert',
    'stdp_m.mod': 'POINT_PROCESS synaptic mechanism — not an ion channel',
    'stdp_ca.mod': 'POINT_PROCESS synaptic mechanism — not an ion channel',
}

# Files whose content is generated fresh (not text-transformed from original)
GENERATED_FILES = {
    'sca.mod': SCA_MOD,
    'kca.mod': KCA_MOD,
    'it2.mod': IT2_MOD,
}

# Files that are text-transformed from their original content
TEXT_TRANSFORMS = {
    'na3.mod': transform_na3,
    'na3dend.mod': transform_na3,
    'na3shifted.mod': transform_na3,
    'kdrca1.mod': transform_kdrca1,
    'kap.mod': transform_kap,
}


def reformat(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    mod_files = [f for f in os.listdir(input_dir) if f.endswith('.mod')]

    converted, skipped, unknown = [], [], []

    for filename in sorted(mod_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if filename in SKIP_FILES:
            print(f"[SKIP]    {filename}: {SKIP_FILES[filename]}")
            skipped.append(filename)
            continue

        if filename in GENERATED_FILES:
            new_content = GENERATED_FILES[filename]
            with open(output_path, 'w') as f:
                f.write(new_content)
            print(f"[REWRITE] {filename}")
            converted.append(filename)
            continue

        if filename in TEXT_TRANSFORMS:
            with open(input_path, 'r') as f:
                content = f.read()
            transform_fn = TEXT_TRANSFORMS[filename]
            new_content = transform_fn(content)
            with open(output_path, 'w') as f:
                f.write(new_content)
            print(f"[TRANSFORM] {filename}")
            converted.append(filename)
            continue

        # Unknown file — copy as-is and warn
        with open(input_path, 'r') as f:
            content = f.read()
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"[COPY]    {filename}: no transformation defined, copied as-is")
        unknown.append(filename)

    print(f"\nDone. Converted: {len(converted)}, Skipped: {len(skipped)}, "
          f"Copied unchanged: {len(unknown)}")
    if skipped:
        print(f"Skipped files: {', '.join(skipped)}")


def main():
    parser = argparse.ArgumentParser(
        description='Reformat Wilmes_2016 MOD files for MODFileConverter compatibility.'
    )
    parser.add_argument(
        '--input-dir',
        default='examples/Wilmes_2016/biophys/mod',
        help='Directory containing original .mod files'
    )
    parser.add_argument(
        '--output-dir',
        default='examples/Wilmes_2016/biophys/tempmod',
        help='Directory to write reformatted .mod files'
    )
    args = parser.parse_args()
    reformat(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
