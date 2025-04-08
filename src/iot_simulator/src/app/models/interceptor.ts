import { FlowGenerator } from "@models/flow-generator";
import { Node } from "@models/node";
import {
    ICMPPacket,
    ICMPType,
    Packet,
    TransportProtocol,
} from "@models/packet";

/**
 * Interceptor de paquetes de red.
 */
export class Interceptor<T extends FlowGenerator> {
    private readonly _node: Node;
    public readonly generator: T;

    /**
     * Crea un interceptador de paquetes de red.
     *
     * @param node Nodo de red.
     */
    public constructor(node: Node, generator: new (node: Node) => T) {
        this._node = node;
        this.generator = new generator(node);
    }

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
                this._node.sendPacket(
                    FlowGenerator.ICMPEchoReply(
                        this._node.ip ?? "",
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
