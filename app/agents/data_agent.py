"""Agent for data wrangling and processing. Or  Making all data Processing Task """
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger , log_execution_time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import path

logger = get_logger(__name__)
