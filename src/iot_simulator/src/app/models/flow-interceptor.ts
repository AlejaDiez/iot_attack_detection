import { Node } from "@models/node";
import {
    ICMPPacket,
    ICMPType,
    Packet,
    TransportProtocol,
} from "@models/packet";

/**
 * Interceptor de flujo.
 */
export class FlowInterceptor {
    /**
     * Crea un generador de flujos de red.
     *
     * @param node Nodo de red.
     */
    public constructor(private readonly node: Node) {}

    /**
     * Intercepta un paquete de red y realiza una acción en caso de ser necesario.
     *
     * @param packet Paquete de red.
     */
    public intercept(packet: Packet): void {
        switch (packet.transportProtocol) {
            case TransportProtocol.ICMP:
                this._interceptICMP(packet as ICMPPacket);
                break;
            case TransportProtocol.TCP:
                break;
            case TransportProtocol.UDP:
                break;
        }
    }

    /**
     * Intercepta un paquete ICMP y realiza una acción en caso de ser necesario.
     *
     * @param packet Paquete ICMP.
     */
    private _interceptICMP(packet: ICMPPacket): void {
        switch (packet.type) {
            case ICMPType.ECHO_REQUEST:
                this.node.sendPacket(
                    Packet.ICMPEchoReply(
                        this.node.ip ?? "",
                        packet.srcIP,
                        packet.identifier,
                        packet.sequence,
                    ),
                );
                break;
            case ICMPType.ECHO_REPLY:
                break;
        }
    }
}
