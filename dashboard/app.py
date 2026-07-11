# This Script is responsisble for the Streamlit Dashboard that actually uses and Initializes the Pipeline Class
# It provides a human readable interface to the user to actually see what the camera is seeing
# It also enables a human to review the Incidents the model requests
# It keeps a Background Thread so that a refresh does not reload the entire system
# Importing Necessary Libraries
import threading, streamlit as st
from typing import Optional
from src.core.config import AUTOREFRESH_INTERVAL_MS, INCIDENT_DIR, REVIEW_TITLE
from src.core.pipeline import Pipeline
from src.core.incidents import IncidentStorage
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

def _init() -> None:
    """
    Initializes the Session state
    Uses a Background thread to prevent session reloading on Page refresh
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_thread" not in st.session_state:
        st.session_state.pipeline_thread = None
    if "stop" not in st.session_state:
        st.session_state.stop = None
    if "incident_storage" not in st.session_state:
        st.session_state.incident_storage = IncidentStorage()
        
def _is_running() -> bool:
    """
    Whether the pipeline is currently running
    asks the background thread itself, rather than a simple flag
    Returns:
        running: A Boolean that is true if pipeline is running else false
    """
    thread: Optional[threading.Thread] = st.session_state.pipeline_thread
    running = thread.is_alive() if thread is not None else False
    return running

def _start() -> None:
    """
    Starts a fresh Monitoring run on a new background thread
    does nothing if the pipeline is already running
    """
    if _is_running():
        return None
    
    if st.session_state.pipeline is None:
        st.session_state.pipeline = Pipeline()
    else:
        st.session_state.pipeline.reset()
        
    stop = threading.Event()
    thread = threading.Thread(target=st.session_state.pipeline.run, kwargs={"should_stop": stop.is_set}, daemon=True)
    thread.start()
    
    st.session_state.pipeline_thread = thread
    st.session_state.stop = stop
    
def _stop() -> None:
    """
    Stops the Running Pipeline thread via the shared threading.Event
    """
    if st.session_state.stop is not None:
        st.session_state.stop.set()
        
def _render_monitor() -> None:
    """
    Renders the Live Monitoring controls like Start, Stop Buttons and an indicator to whether it is running
    """
    st.subheader("Live Monitoring Camera")
    
    running = _is_running()
    status = "Running" if running else "Stopped"
    st.markdown(f"**Status:** {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring", disabled=running, use_container_width=True):
            _start()
            st.rerun()
    with col2:
        if st.button("Stop Monitoring", disabled=not running, use_container_width=True):
            _stop()
            st.rerun()
            
    if running:
        frame, count = st.session_state.pipeline.get_latest_frame()
        if frame is not None:
            st.image(frame[:, :, ::-1], use_container_width=True)
            st.caption(f"Tracked: {count}")
        else:
            st.caption("Waiting for First Frame")
            
def _render_incident(record: dict) -> None:
    """
    Renders a Single Incident as a Card, its data and its video clip along with buttons for Approve and Reject
    Args:
        record: A Dictionary containing the Info of a Single Incident
    """
    store: IncidentStorage = st.session_state.incident_storage
    
    with st.container(border=True):
        st.markdown(f"**{record['id']}** — {record['object']} ({record['confidence']:.0%} confidence) — status: `{record['status']}`")
        st.caption(f"{record['time']} · camera: {record['cam_id']}")
        
        path = Path(INCIDENT_DIR) / record['video']
        if path.exists():
            st.video(str(path))
        else:
            st.caption("Video not Avaliable")
            
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Approve", key=f"approve_{record['id']}", use_container_width=True):
                store.update(id=record['id'], status="approved")
                st.rerun()
        with col2:
            if st.button("Reject", key=f"reject_{record['id']}", use_container_width=True):
                store.update(id=record["id"], status="rejected")
                st.rerun()
                
def _render_review() -> None:
    """
    Renders the Reviewbale Incident List and a status filter so reviewer can sort by pending, accepted or rejected
    """
    st.subheader("Incident Review")
    store: IncidentStorage = st.session_state.incident_storage
    
    status = st.radio("Filter by Status", options=["pending", "approved", "rejected", "all"])
    filter = None if status == "all" else status
    records = store.list_all(status=filter)
    
    if not records:
        st.info("No Incidents Matching the Given Filter")
        
    for record in records:
        _render_incident(record=record)
        
def main() -> None:
    """
    The Main entry Point for the Dashboard and the final function that wires everything together
    ran using streamlit run app.py (when in cleanstreets.ai/dashboard)
    """
    st.set_page_config(page_title=REVIEW_TITLE, layout="wide")
    st.title(REVIEW_TITLE)
    
    _init() 
    st_autorefresh(interval=AUTOREFRESH_INTERVAL_MS, key="dashboard_autorefresh")
    
    _render_monitor()
    st.divider()
    _render_review()
    
if __name__ == "__main__":
    main()