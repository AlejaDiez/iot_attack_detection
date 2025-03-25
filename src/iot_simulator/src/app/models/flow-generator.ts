import { Node } from "@models/node";
import { ICMPPacket, ICMPType, TransportProtocol } from "@models/packet";

/**
 * Generador de flujos de red.
 */
export class FlowGenerator {
    /** Nodo de red */
    protected readonly node: Node;

    /**
     * Crea un generador de flujos de red.
     *
     * @param node Nodo de red.
     */
    public constructor(node: Node) {
        this.node = node;
    }

    ////////////////////////////
    // SIMULACIÓN DE COMANDOS //
    ////////////////////////////

    /**
     * Realiza un ping a una dirección IP.
     *
     * @param dstIP Dirección IP de destino.
     */
    public ping(dstIP: string): void {
        const packet = FlowGenerator.ICMPEchoRequest(this.node.ip ?? "", dstIP);

        this.node.sendPacket(packet);
    }

    /////////////////////////////
    // GENERADORES DE PAQUETES //
    /////////////////////////////

    /**
     * Crea un paquete ICMP Echo Request.
     *
     * @param srcIP Dirección IP de origen.
     * @param dstIP Dirección IP de destino.
     * @returns Paquete ICMP Echo Request.
     */
    public static ICMPEchoRequest(srcIP: string, dstIP: string): ICMPPacket {
        const payload = "Ping request";
        const headerSize = 8;
        const payloadSize = new TextEncoder().encode(payload).length;

        return {
            srcIP,
            dstIP,
            transportProtocol: TransportProtocol.ICMP,
            payload,
            totalBytes: headerSize + payloadSize,
            headerSize,
            payloadSize,
            timestamp: new Date(),
            ttl: 64,
            type: ICMPType.ECHO_REQUEST,
            code: 0,
            identifier: Math.floor(Math.random() * 65535),
            sequence: Math.floor(Math.random() * 65535),
        };
    }

    /**
     * Crea un paquete ICMP Echo Reply.
     *
     * @returns Paquete ICMP Echo Reply.
     */
    public static ICMPEchoReply(
        srcIP: string,
        dstIP: string,
        identifier: number,
        sequence: number,
    ): ICMPPacket {
        const payload = "Pong response";
        const headerSize = 8;
        const payloadSize = new TextEncoder().encode(payload).length;

        return {
            srcIP,
            dstIP,
            transportProtocol: TransportProtocol.ICMP,
            payload,
            totalBytes: headerSize + payloadSize,
            headerSize,
            payloadSize,
            timestamp: new Date(),
            ttl: 64,
            type: ICMPType.ECHO_REPLY,
            code: 0,
            identifier,
            sequence,
        };
    }
}
