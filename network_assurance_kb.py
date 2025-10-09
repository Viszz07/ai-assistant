NETWORK_ASSURANCE_KNOWLEDGE = {
    "service_flow": {
        "boot_initialization": [
            "main.py starts the assurance service",
            "config_loader.py reads configuration and thresholds",
            "resource_manager.py allocates compute resources via VIM",
            "vnfm_client.py deploys CNFs/VNFs",
            "metrics_collector.py starts KPI subscriptions"
        ],
        "active_monitoring": [
            "data_ingestor.py collects telemetry data",
            "anomaly_detector.py identifies metric deviations",
            "alarm_manager.py raises alarms and manages incidents",
            "policy_enforcer.py applies scaling and throttling policies"
        ],
        "recovery_optimization": [
            "remediation_engine.py handles auto-recovery actions",
            "report_generator.py creates incident summaries",
            "audit_logger.py maintains compliance records"
        ]
    },
    "network_components": {
        "4g_core": {
            "components": ["MME", "SGW", "PGW", "HSS", "PCRF", "VNFM", "VIM"],
            "description": "4G LTE Core network components responsible for mobility, session management, and policy control"
        },
        "5g_ran": {
            "components": ["AMF", "SMF", "UPF", "DU", "CU", "VNFM", "VIM", "NWDAF", "PCF"],
            "description": "5G Radio Access Network components handling access, mobility, and data forwarding"
        }
    },
    "component_details": {
        "MME": "Mobility Management Entity - handles UE authentication, mobility, and bearer establishment",
        "SGW": "Serving Gateway - routes data packets and manages user plane tunneling",
        "PGW": "PDN Gateway - provides connectivity to external networks and IP address allocation",
        "HSS": "Home Subscriber Server - stores user profiles and authentication data",
        "PCRF": "Policy and Charging Rules Function - manages QoS policies and charging rules",
        "AMF": "Access and Mobility Management Function - handles UE registration and mobility",
        "SMF": "Session Management Function - manages PDU sessions and IP address allocation",
        "UPF": "User Plane Function - forwards user data packets between RAN and external networks",
        "DU": "Distributed Unit - handles lower layer radio protocols and scheduling",
        "CU": "Centralized Unit - manages radio resource control and higher layer protocols",
        "NWDAF": "Network Data Analytics Function - provides network insights and predictions",
        "PCF": "Policy Control Function - manages network policies and slicing",
        "VNFM": "Virtual Network Function Manager - manages VNF lifecycle",
        "VIM": "Virtual Infrastructure Manager - manages compute, storage, and network resources",
        "VNF": "Virtual Network Function - software implementation of network functions that can be deployed on virtual machines",
        "CNF": "Container Network Function - network function packaged as a container for cloud-native deployment"
    },
    "common_issues": {
        "latency": {
            "causes": ["network congestion", "resource overload", "configuration issues"],
            "indicators": ["high RTT", "packet loss", "queue buildup"],
            "solutions": ["scale resources", "optimize routing", "check fiber links"]
        },
        "packet_loss": {
            "causes": ["link failures", "buffer overflows", "interference"],
            "indicators": ["CRC errors", "retransmission rates", "SLA violations"],
            "solutions": ["check physical links", "adjust buffer sizes", "reroute traffic"]
        },
        "connectivity": {
            "causes": ["authentication failures", "network partitions", "resource exhaustion"],
            "indicators": ["connection timeouts", "registration failures", "session drops"],
            "solutions": ["verify credentials", "check network topology", "increase capacity"]
        }
    },
    "metrics": {
        "performance": ["latency", "throughput", "packet_loss", "jitter"],
        "reliability": ["availability", "MTTR", "error_rate"],
        "efficiency": ["resource_utilization", "energy_consumption", "cost_effectiveness"]
    }
}

def get_component_context(component_type):
    """Get detailed context about a specific network component"""
    component_type = component_type.upper()
    return NETWORK_ASSURANCE_KNOWLEDGE["component_details"].get(
        component_type,
        f"Component {component_type} not found in network architecture knowledge base"
    )

def get_service_flow_context(phase=None):
    """Get service flow context for a specific phase or all phases"""
    if phase:
        phase_lower = phase.lower().replace("/", "_")
        return NETWORK_ASSURANCE_KNOWLEDGE["service_flow"].get(
            phase_lower,
            f"Phase {phase} not found in service flow"
        )
    return NETWORK_ASSURANCE_KNOWLEDGE["service_flow"]

def analyze_query_intent(query):
    """Analyze user query to determine if it's about logs, general knowledge, or flow"""
    query_lower = query.lower()

    log_keywords = ["error", "warning", "log", "logs", "entries", "messages", "alarms"]
    flow_keywords = ["flow", "sequence", "process", "steps", "how does", "boot", "initialization"]
    component_keywords = ["component", "MME", "SGW", "AMF", "SMF", "what is", "explain"]

    log_score = sum(1 for keyword in log_keywords if keyword in query_lower)
    flow_score = sum(1 for keyword in flow_keywords if keyword in query_lower)
    component_score = sum(1 for keyword in component_keywords if keyword in query_lower)

    if log_score >= flow_score and log_score >= component_score:
        return "log_analysis"
    elif flow_score >= component_score:
        return "flow_explanation"
    else:
        return "component_explanation"
