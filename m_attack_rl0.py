from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import arp

from ryu.lib import hub
import csv
import time
import math
import statistics

from svm import SVM
from rl_model import QLearningAgent


APP_TYPE = 1
#0 datacollection, 1 ddos detection

PREVENTION = 1
# ddos prevention

#TEST_TYPE is applicable only for data collection
#0  normal traffic, 1 attack traffic
TEST_TYPE = 1

#data collection time interval in seconds
INTERVAL = 3
#-------------------------------------------------------#



gflows = []


old_ssip_len = 0
prev_flow_count = 0

FLOW_SERIAL_NO = 0
iteration = 0




def get_flow_number():
    global FLOW_SERIAL_NO
    FLOW_SERIAL_NO = FLOW_SERIAL_NO + 1
    return FLOW_SERIAL_NO


def init_portcsv(dpid):
    fname = "switch_" + str(dpid) + "_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "sfe","ssip","rfip","type"]
    writ.writerow(header)


def init_flowcountcsv(dpid):
    fname = "switch_" + str(dpid) + "_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "flowcount"]
    writ.writerow(header)



def update_flowcountcsv(dpid, row):
    fname = "switch_" + str(dpid) + "_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    writ.writerow(row)


def update_portcsv(dpid, row):
    fname = "switch_" + str(dpid) + "_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)


def update_resultcsv(row):
    fname = "result.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)





class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_thread = hub.spawn(self._flow_monitor)
        self.datapaths = {}
        self.mitigation = 0
        self.svmobj = None
        self.arp_ip_to_port = {}

        if APP_TYPE == 1:
            #self.svmobj = SVM()
            self.agent = QLearningAgent(actions=[0, 1], state_space_size=10000)  # Adjust state space size as needed

        # Initialize state variables
        self.current_state = 0
        self.previous_state = 0
        self.previous_action = 0

    def _flow_monitor(self):
        #inital delay
        hub.sleep(5)
        while True:
            #self.logger.info("Starts Flow monitoring")
            for dp in self.datapaths.values():
                self.request_flow_metrics(dp)
            hub.sleep(INTERVAL)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath
        #init_portcsv(datapath.id)


        flow_serial_no = get_flow_number()

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, flow_serial_no)

        init_portcsv(datapath.id)
        init_flowcountcsv(datapath.id)

    def request_flow_metrics(self, datapath):
        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)


    def _speed_of_flow_entries(self, flows):
        global prev_flow_count
        curr_flow_count = 0
        #collect the packet_count from all the flows
        for flow in flows:
            curr_flow_count += 1

        #print "speed of flow entries ", flow_count
        sfe = curr_flow_count - prev_flow_count
        prev_flow_count = curr_flow_count
        return sfe


    def _speed_of_source_ip(self, flows):
        global old_ssip_len
        ssip = []
        #print "length of flow table " ,len(flows)
        for flow in flows:
            m = {}
            for i in flow.match.items():
                key = list(i)[0]  # match key
                val = list(i)[1]  # match value
                if key == "ipv4_src":
                    #print key,val
                    if val not in ssip:
                        ssip.append(val)
        #print "source_ips ", ssip
        cur_ssip_len = len(ssip)
        ssip_result = cur_ssip_len - old_ssip_len
        old_ssip_len = cur_ssip_len
        #print "ssip ", ssip
        return ssip_result


    def _ratio_of_flowpair(self, flows):
        #find total number of flows
        # find collaborative flows (ideal case - all - 1 )
        flow_count = 0
        for flow in flows:
            flow_count += 1
        #print "total number of flows ", flow_count
        #excluding the table miss entry from flow count
        flow_count -= 1

        collaborative_flows = {}
        for flow in flows:
            m = {}
            srcip = dstip = None
            for i in flow.match.items():
                key = list(i)[0]  # match key
                val = list(i)[1]  # match value
                if key == "ipv4_src":
                    srcip = val
                    #print key,val
                if key == "ipv4_dst":
                    dstip = val
            if srcip and dstip:
                fwdflowhash = srcip + "_" + dstip
                revflowhash = dstip + "_" + srcip
                #check flowhash is already exist
                if not fwdflowhash in collaborative_flows:
                    #check you have reverse flowhash exists?
                    if not revflowhash in collaborative_flows:
                        collaborative_flows[fwdflowhash] = {}
                    else:
                        collaborative_flows[revflowhash][fwdflowhash] = 1
        #identify number of collaborative flows
        onesideflow = iflow = 0
        for key in collaborative_flows:
            if collaborative_flows[key] == {}:
                onesideflow += 1
            else:
                iflow +=2
        #print "collaborative_flows", collaborative_flows
        #print "oneside flow", onesideflow
        #print "collaborative flow ", iflow
        if flow_count != 0 :
            rfip = float(iflow) / flow_count
            #print "rfip ", rfip
            return rfip
        return 1.0

    @set_ev_cls([ofp_event.EventOFPFlowStatsReply], MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        global gflows, iteration
        t_flows = ev.msg.body
        flags = ev.msg.flags
        dpid = ev.msg.datapath.id
        gflows.extend(t_flows)

        if flags == 0:
            sfe  = self._speed_of_flow_entries(gflows)
            ssip = self._speed_of_source_ip(gflows)
            rfip = self._ratio_of_flowpair(gflows)

            if APP_TYPE == 1:
                
                # Predict action using RL model
                state = self._get_state(sfe, ssip, rfip)
                action= self.agent.choose_action(state)
                
                if action == 1:  # Block traffic
                    print("Attack traffic detected. Mitigation applied.")
                    self.mitigation = 1
                    if PREVENTION == 1:
                        print("Mitigation active: Blocking ports.")
                        #self.block_port(dpid)
                else:
                    print("Normal traffic. No mitigation required.")
                reward = self._calculate_reward(action)
                self.agent.learn(self.previous_state, self.previous_action, reward, state)

                # Update previous state and action
                self.previous_state = state
                self.previous_action = action        
                gflows = []

            else:
                t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
                row = [t, str(sfe), str(ssip), str(rfip)]
                self.logger.info(row)

                update_portcsv(dpid, row)
                update_resultcsv([str(sfe), str(ssip), str(rfip)])
            gflows = []


            #update the flowcount csv file
            t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
            update_flowcountcsv(dpid, [t, str(prev_flow_count)])
        gflows = []
    
    def _get_state(self, sfe, ssip, rfip):
        # Map the flow statistics to a state representation (e.g., a number)
        # For simplicity, here we just use a simple function of sfe, ssip, and rfip
        # You may want to use binning or more sophisticated techniques
        sfe_norm = min(sfe / 10, 1.0)
        ssip_norm = min(ssip / 5, 1.0)
        rfip_norm = min(rfip / 0.1, 1.0)
        # Convert the continuous values to integers by multiplying by 100 or 10 and rounding them
        state = int(sfe_norm * 100 + ssip_norm * 10 + rfip_norm)
        return state
        #return (sfe_norm, ssip_norm, rfip_norm)


    def _calculate_reward(self, action):
        # Define your reward function based on the action
        # Example: if action 1 (mitigation) was taken and attack was detected, reward is high
        if action == 1 and self.mitigation:
            return 1  # Reward for successful mitigation
        elif action == 0 and not self.mitigation:
            return 1  # Reward for correctly allowing normal traffic
        else:
            return -1  # Penalize for wrong action

    def add_flow(self, datapath, priority, match, actions,serial_no, buffer_id=None, idletime=0, hardtime=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, buffer_id=buffer_id,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, priority=priority,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)


    def block_port(self, datapath, portnumber, dpid):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        actions = []
        flow_serial_no = get_flow_number()
        self.add_flow(datapath, 100, match, actions, flow_serial_no, hardtime=120)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:

            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})
        self.arp_ip_to_port[dpid].setdefault(in_port, [])
        #self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        #if ARP Request packet , log the IP and MAC Address from that port
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            #self.logger.info("Received ARP Packet %s %s %s ", dpid, src, dst)
            a = pkt.get_protocol(arp.arp)
            #print "arp packet ", a
            if a.opcode == arp.ARP_REQUEST or a.opcode == arp.ARP_REPLY:
                if not a.src_ip in self.arp_ip_to_port[dpid][in_port]:
                    self.arp_ip_to_port[dpid][in_port].append(a.src_ip)
                    #print "arp_ip_to_port " ,self.arp_ip_to_port


        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:

            # check IP Protocol and create a match for IP
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto


                if self.mitigation and PREVENTION:
                    if not (srcip in self.arp_ip_to_port[dpid][in_port]):
                        print("attack detected from port ", in_port)
                        print("Block the port ", in_port)
                        self.block_port(datapath, in_port, dpid)
                        print(ip)
                        return

                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=srcip, ipv4_dst=dstip)

                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out
                flow_serial_no = get_flow_number()
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no,  buffer_id=msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
