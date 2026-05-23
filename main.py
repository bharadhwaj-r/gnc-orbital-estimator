import streamlit as st
import plotly.graph_objects as go
import numpy as np

from orbital_mechanics import propogate_orbit, R_Earth
from kalman_filter import generate_noisy_measurements, KalmanFilter

# --- UI CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title='GNC Sandbox', layout='wide', page_icon='🚀')

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-weight: 600; color: #E2E8F0; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .stMetric label { font-family: 'JetBrains Mono', monospace; color: #94A3B8 !important; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'sim_data_exists' not in st.session_state:
    st.session_state.sim_data_exists = False

# --- HEADER & MISSION BRIEFING ---
st.title("🚀 GNC Sandbox: Orbital Estimation")
st.markdown("---")

header_col1, header_col2 = st.columns([1, 1])

with header_col1:
    st.markdown("""
    ### 📡 Mission Briefing & Core Capabilities
    Spacecraft rely on hardware sensors to determine their position, but hardware isn't perfect as it can produce **noisy, inaccurate data**. If a flight computer blindly were to blindly trust this data, the mission can fail.
    
    **What this sandbox enables:**
    * **For Students:** Visually demonstrates the mathematical tug-of-war between theoretical orbital mechanics and real-world hardware limitations.
    * **For Professionals:** Provides a high-fidelity, RK45-integrated testing environment to benchmark Guidance, Navigation, and Control (GNC) estimation algorithms against configurable sensor degradation profiles.
    
    **Telemetry Legend:**
    * <span style='color:#00ffcc; font-weight:bold;'>Cyan Line:</span> The mathematically perfect, true orbit.
    * <span style='color:#ff3333; font-weight:bold;'>Red Dots:</span> The raw, noisy sensor data.
    * <span style='color:#ffcc00; font-weight:bold;'>Yellow Line:</span> The Kalman Filter's mathematical estimation.
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    ### 🛠️ Flight Director's Guide
    **How to use this tool:**
    1. **Set Mission Parameters:** Open the sidebar to define your target Low Earth Orbit (LEO) altitude and simulation time.
    2. **Configure Hardware Quality:** Adjust the **Sensor Noise** slider to simulate anything from precision star trackers to cheap, commercial components.
    3. **Launch the Simulation:** Run the RK45 physics engine and the Kalman Filter sequentially to generate telemetry.
    4. **Analyze the Data:** Use the **Visibility Toggles** in the sidebar to isolate specific data layers in the 3D viewer. 
    """)
# --- KALMAN FILTER MATHEMATICAL BREAKDOWN ---
st.markdown("### 🧮 The Mathematics of Estimation: The Kalman Filter")
math_col1, math_col2 = st.columns([1, 1])

with math_col1:
    st.markdown("""
    At its core, a Kalman Filter is an optimal estimation algorithm. It doesn't just smooth data; it runs a continuous, real-time cycle of **prediction** and **correction** by combining a physical model with incoming sensor measurements.
    
    The filter maintains a state vector $\hat{x}$ (position and velocity) and an uncertainty covariance matrix $P$ (how much we trust our current estimate). 
    
    **Step 1: The Prediction Phase (Physics Engine)**
    Before a sensor even takes a reading, the flight computer propagates the state forward using our structural transition matrix $F$ (derived from our orbital dynamics equations):
    """)
    st.latex(r"\hat{x}_{k \mid k-1} = F_k \hat{x}_{k-1 \mid k-1}")
    st.latex(r"P_{k \mid k-1} = F_k P_{k-1 \mid k-1} F_k^T + Q_k")
    st.markdown("Where $Q$ represents **Process Noise**—the physical uncertainties like atmospheric drag or unmodeled gravitational anomalies.")

with math_col2:
    st.markdown("""
    **Step 2: The Correction Phase (Sensor Measurement)**
    When a noisy hardware sensor delivers a telemetry packet $z_k$, the filter calculates the **Kalman Gain ($K$)**. This value acts as a mathematical scale between 0 and 1 to determine who to trust more: our physics model or our hardware.
    """)
    st.latex(r"K_k = P_{k \mid k-1} H_k^T (H_k P_{k \mid k-1} H_k^T + R_k)^{-1}")
    st.markdown("""
    Where $R$ is the **Measurement Noise Covariance** (configured by your sidebar noise slider). 
    * If sensor noise ($R$) is high, $K$ drops, and the filter heavily trusts the **physics engine**.
    * If sensor noise ($R$) is low, $K$ increases, and the filter heavily trusts the **hardware**.
    
    Finally, the state is updated to yield the optimal estimate:
    """)
    st.latex(r"\hat{x}_{k \mid k} = \hat{x}_{k \mid k-1} + K_k (z_k - H_k \hat{x}_{k \mid k-1})")

st.markdown("---")


# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Mission Parameters")
altitude = st.sidebar.slider("Starting Altitude (km)", min_value=200, max_value=2000, value=400, step=50)
sim_time_hours = st.sidebar.slider("Simulation Time (hours)", min_value=0.5, max_value=12.0, value=1.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📡 Sensor Hardware")
noise_level = st.sidebar.slider("Sensor Noise (km)", min_value=0, max_value=1000, value=400, step=50, help="Simulates hardware inaccuracy.")

st.sidebar.markdown("---")
st.sidebar.header("👁️ Visibility Toggles")
show_true = st.sidebar.checkbox("Show True Trajectory (Cyan)", value=True)
show_noise = st.sidebar.checkbox("Show Sensor Noise (Red)", value=True)
show_kf = st.sidebar.checkbox("Show Kalman Filter (Yellow)", value=True)

t_span = sim_time_hours * 3600

# --- SIMULATION TRIGGER ---
if st.sidebar.button("Launch Simulation 🚀", use_container_width=True, type="primary"):
    with st.spinner("Calculating Orbital Dynamics & Filtering Telemetry..."):
        # Calculate dynamic circular velocity
        mu = 398600.0 # Earth's gravitational parameter (km^3/s^2)
        radius = R_Earth + altitude
        v_mag = np.sqrt(mu / radius)
        
        r0 = np.array([radius, 0.0, 0.0]) 
        v0 = np.array([0.0, v_mag, 0.0]) # Velocity now changes based on altitude!
        
        # Save the exact velocity to session state so the UI can display it
        st.session_state.v_mag = v_mag
        
        # 1. Physics Engine
        times, states = propogate_orbit(r0, v0, t_span)
        st.session_state.x_vals, st.session_state.y_vals, st.session_state.z_vals = states[0, :], states[1, :], states[2, :]
        
        # 2. Sensor Simulation
        noisy_data = generate_noisy_measurements(states, noise_std_dev=noise_level)
        st.session_state.noisy_x, st.session_state.noisy_y, st.session_state.noisy_z = noisy_data[0, :], noisy_data[1, :], noisy_data[2, :]
        
        # 3. Kalman Filter
        dt = t_span / len(times)
        kf = KalmanFilter(initial_state=np.concatenate((r0, v0)), noise_std_dev=noise_level, dt=dt)
        
        filtered_x, filtered_y, filtered_z = [], [], []
        for i in range(len(times)):
            kf.predict()
            current_measurement = np.array([st.session_state.noisy_x[i], st.session_state.noisy_y[i], st.session_state.noisy_z[i]])
            filtered_pos = kf.update(current_measurement)
            filtered_x.append(filtered_pos[0])
            filtered_y.append(filtered_pos[1])
            filtered_z.append(filtered_pos[2])
            
        st.session_state.filtered_x, st.session_state.filtered_y, st.session_state.filtered_z = filtered_x, filtered_y, filtered_z
        
        # Save metrics
        st.session_state.last_altitude = altitude
        st.session_state.last_noise = noise_level
        st.session_state.packets = len(times)
        st.session_state.sim_data_exists = True

# --- RENDER DASHBOARD (Only if data exists in session state) ---
if st.session_state.sim_data_exists:
    st.markdown("### 📊 Live Telemetry Feed")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Altitude", f"{st.session_state.last_altitude} km")
    col2.metric("Orbital Velocity", f"{st.session_state.v_mag:.2f} km/s")
    col3.metric("Sensor Variance", f"± {st.session_state.last_noise} km")
    col4.metric("Telemetry Packets", f"{st.session_state.packets}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Earth Wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = R_Earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_Earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_Earth * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', opacity=0.1, showscale=False,
        contours=dict(x=dict(show=True, color='#1E40AF', width=1), 
                      y=dict(show=True, color='#1E40AF', width=1), 
                      z=dict(show=True, color='#1E40AF', width=1)),
        name='Earth'
    ))
    
    # TRACE 1: Perfect Orbit
    if show_true:
        fig.add_trace(go.Scatter3d(
            x=st.session_state.x_vals, y=st.session_state.y_vals, z=st.session_state.z_vals,
            mode='lines', line=dict(color='#00ffcc', width=6),
            name='True Trajectory'
        ))
    
    # TRACE 2: Noisy Sensors
    if show_noise and st.session_state.last_noise > 0:
        fig.add_trace(go.Scatter3d(
            x=st.session_state.noisy_x, y=st.session_state.noisy_y, z=st.session_state.noisy_z,
            mode='markers', marker=dict(size=3, color='#ff3333', opacity=0.7),
            name='Raw Sensor Noise'
        ))
        
    # TRACE 3: Kalman Filter
    if show_kf:
        fig.add_trace(go.Scatter3d(
            x=st.session_state.filtered_x, y=st.session_state.filtered_y, z=st.session_state.filtered_z,
            mode='lines', line=dict(color='#ffcc00', width=4),
            name='Kalman Filter Estimate'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (km)', gridcolor='#334155', showbackground=False, zerolinecolor='#475569'),
            yaxis=dict(title='Y (km)', gridcolor='#334155', showbackground=False, zerolinecolor='#475569'),
            zaxis=dict(title='Z (km)', gridcolor='#334155', showbackground=False, zerolinecolor='#475569'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='#0F172A', plot_bgcolor='#0F172A',
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="white"))
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Set your mission parameters in the sidebar and click **Launch Simulation** to begin.")