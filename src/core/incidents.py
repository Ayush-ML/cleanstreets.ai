# This Script is responsible for the reading, writing and updating Incident Records
# IT acts as a connection from the pipeline (src/core/pipeline.py) to the Streamlit Dashboard (dashboard/app.py)
# the Functions in this script like write helps the Reviewer to Write a Incident to disk with their final decision
# The list_all function helps the Dashboard to get all pending Requests from the pipeline
# There are also some other functions that are mentioned in the code below
# Importing Necessary Libraries
import json, os, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from src.core.config import DEFAULT_CAMERA, INCIDENT_DIR, INCIDENTS_FILE, VALID_STATUS
from src.core.utils import Incident
from src.core.utils import IncidentDict

class IncidentStorage:
    """
    Reads and Writes incident records to a Single JSON File on the Disk
    Uses a Threading Lock so that it can run in the background
    The Following id the shape of one incident record:
        {
            "id": "inc_0001",                 
            "time": "2026-07-06T14:32:10+00:00", 
            "cam_id": "cam1",
            "pid": 3,                    
            "oid": 7,                   
            "object": "bottle",            
            "confidence": 0.81,
            "frame_n": 452,
            "video": "inc_0001.mp4",  
            "status": "pending",              
            "notes": ""                        
        }
    """
    
    def __init__(self) -> None:
        """
        Sets up the storage which points at the given JSON File Path
        Garuntees that the File and Parent Directories Exist for use in Further Functions
        """
        self.path = Path(INCIDENTS_FILE)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_all(records=[])
        
    def _read_all(self) -> List[IncidentDict]:
        """
        Reads and Parses the Entire Incidents JSON File into a List of Dictionaries for easy use in further Functions
        Returns:
            records: A List of Dictionaries containing Data about all Incident Record
        """
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Receieved Exception from reading Incidents JSON File, {e}")
            return []
        
    def _write_all(self, records: List[IncidentDict]) -> None:
        """
        Writes the Full List of Records to the Incidents JSON File using an 'ATMOIC' Method
        This means that the Record is first written to a Temporary File beofre writing to the Main File
        This Prevents A Corrupted Record being Stored due to Any reason it may be
        """
        temp = self.path.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=4)
        os.replace(temp, self.path)
        
    def _get_next_id(self) -> str:
        """
        Generates the Next Incident ID that is to be used as the filename in the Incident Storage
        It scans All Files in the Directory and sorts by Latest, If no Files Exist, Return First One,
        else return the ID Number of the Latest File + 1
        Returns:
            next_id: The Filename as a String for the next Incident in the Directory
        """
        existing_ids = sorted(p.stem for p in Path(INCIDENT_DIR).glob("inc_*.*") if p.stem.startswith("inc_") and p.stem[len("inc_"):].isdigit())
        if not existing_ids:
            return "inc_0001"
        else:
            latest = int(existing_ids[-1][len("inc_"):])
            return f"inc_{latest + 1:04d}"
        
    def save(self, incident: Incident, camera_id: str = DEFAULT_CAMERA) -> IncidentDict:
        """
        Saves a New Incident as a Permanent Record on the Incidents JSON File
        Uses The Format written in the Doctring of __init__
        Args:
            incident: The Incident Record containing all the Information
            camera_id: An Optional Field incase there are multiple cameras, Dont even know why i added this
        Returns:
            record: The Full Record Dict that is parsed form the Incident Record given as input
        """
        with self._lock:
            records = self._read_all()
            new_id = self._get_next_id()
            video = f"{new_id}.mp4"
            
            record = {
                "id": new_id,
                "time": datetime.now(timezone.utc).isoformat(),
                "cam_id": camera_id,
                "pid": incident.pid,
                "oid": incident.obj_id,
                "object": incident.class_name,
                "confidence": incident.confidence,
                "frame_n": incident.frame_n,
                "video": video,
                "status": "pending",
                "notes": ""
            }
            
            records.append(record)
            self._write_all(records=records)
            
            return record
        
    def list_all(self, status: Optional[str] = None) -> List[IncidentDict]:
        """
        Returns all The Records Sorted with the key as Timestamp in Reverse
        given a status, returns records only for that status
        Args:
            status: An Optional Arguement to only return Results with the Given Status
        Returns:
            records: A List of Incident Records that match the status, if status is not given, returns all Incident Records sorted by Time
        """ 
        with self._lock:
            records = self._read_all()
            
        if status:
            records =  [r for r in records if r['status'] == status]
            
        records.sort(key=lambda x: x['time'], reverse=True)
        return records
    
    def fetch(self, id: str) -> IncidentDict | None:
        """
        Fetches a Single Incident Record based on Its ID
        Args:
            id: The ID matching the Video to search for, The ID is basically just 'inc_' and a number in ascending order
        Returns:
            record: The JSON of the Incident Record matching the Given ID or None if no matches exist
        """
        with self._lock:
            records = self._read_all()
            
        for record in records:
            if record['id'] == id:
                return record
        else:
            return None
        
    def update(self, id: str, status: str, notes: Optional[str] = None) -> bool:
        """
        Updates a Single Incidents Review Status from pending to whatever the Reviewer decides and optionally writes a note from the reviewer itself
        potentially pointing out The AI's Mistakes in Classification
        Called By the Streamlit Dashboard (app.py)
        Args:
            id: The ID of the Incident Record to update the Status for
            status: The New Status the Reviewer must be either "approved or "rejected"
            notes: Optional Notes Written by the Reviewer
        Returns:
            success: A Boolen containing whether the Updating Failed or not, idk why it would fail but i guess ill include this cause claude told me it would be a good decision, may remove the returning all together later
        """
        if status not in VALID_STATUS:
            return False
        with self._lock:
            records = self._read_all()
            
            for record in records:
                if record['id'] == id:
                    record['status'] = status
                    if notes:
                        record['notes'] = notes
                    self._write_all(records=records)
                    return True
            else:
                return False