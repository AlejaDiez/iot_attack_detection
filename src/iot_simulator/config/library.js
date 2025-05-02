/**
 * Function without any prefix is a "private" function.
 *
 * @param  {...any} args - Arguments that are going to be used in the function
 * @returns {void}
 */
function hiddenFunction(...args) {
    // Don't worry, this function is invisible to the user, keep coding!
}

/**
 * This function with intcp name or intcp_ prefix is used to intercept packets when the device receives them.
 * It can be used to log the packets or modify them before they are sent to the target.
 * It can be possible to set different interceptors for different devices:
 *      - intcp: intercept packets from all devices
 *      - intcp_router: intercept packets from routers
 *      - intcp_iot: intercept packets from IoT devices
 *      - intcp_computer: intercept packets from computers
 *
 * @param {Object} self - The attacker device
 * @param {string} self.mac - MAC address of the attacker device
 * @param {string} self.ip - IP address of the attacker device
 * @param {string} self.name - Name of the attacker device
 * @param {string} self.type - Type of the attacker device
 * @param {function(packet) => void} self.send - Function to send a packet
 * @param {Object} packet - The packet that is being intercepted
 * @param {string} packet.srcIP - Source IP address of the packet
 * @param {number} [packet.srcPort] - Source port of the packet
 * @param {string} packet.dstIP - Destination IP address of the packet
 * @param {number} [packet.dstPort] - Destination port of the packet
 * @param {TransportProtocol} packet.transportProtocol - Transport protocol of the packet
 * @param {ApplicationProtocol} [packet.applicationProtocol] - Application protocol of the packet
 * @param {string} [packet.payload] - Payload of the packet
 * @param {number} packet.totalBytes - Total bytes of the packet
 * @param {number} packet.headerSize - Header size of the packet
 * @param {number} packet.payloadSize - Payload size of the packet
 * @param {Date} packet.timestamp - Timestamp of the packet
 * @param {number} packet.ttl - Time to live of the packet
 * @param {number} packet.type - ICMP message type (ECHO_REPLY = 0, ECHO_REQUEST = 8)
 * @param {number} packet.code - ICMP message code (0 = no code)
 * @param {number} packet.identifier - ICMP message identifier
 * @param {number} packet.sequence - ICMP message sequence number
 * @param {number} packet.tcpFlags - TCP flags (FIN = 1, SYN = 2, RST = 4, PSH = 8, ACK = 16, URG = 32)
 * @return {void} - If you don't want to run the default interceptor of the simulator, return null or other value
 */
function intcp(self, packet) {
    // Hey there is a packet here!
    // You can modify or log the packet before sending it to the target
    
    // If you don't want to run the default interceptor of the simulator, you can return null or other value
    // return null;
}

/**
 * This function with cmd_ prefix is used to simulate a command, in the simulator it will show as "ping"
 *
 * @param {Object} self - The attacker device
 * @param {string} self.mac - MAC address of the attacker device
 * @param {string} self.ip - IP address of the attacker device
 * @param {string} self.name - Name of the attacker device
 * @param {string} self.type - Type of the attacker device
 * @param {function(packet) => void} self.send - Function to send a packet
 * @param {string} target - IP addresses of the target device
 * @returns {void}
 */
function cmd_ping(self, target) {
    // Program your command here...
}

/**
 * This function with atk_ prefix is used to attack a device, in the simulator it will show as "DDoS v1"
 *
 * @param {Object} self - The attacker device
 * @param {string} self.mac - MAC address of the attacker device
 * @param {string} self.ip - IP address of the attacker device
 * @param {string} self.name - Name of the attacker device
 * @param {string} self.type - Type of the attacker device
 * @param {function(packet) => void} self.send - Function to send a packet
 * @param {...string} targets - IP addresses of the target devices
 * @returns {void}
 */
function atk_DDoS_v1(self, ...targets) {
    // Program your attack here...
}
