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
    /** Interceptor externo */
    private _externalInterceptor?: (packet: Packet) => void;

    /**
     * Crea un generador de flujos de red.
     *
     * @param node Nodo de red.
     */
    public constructor(private readonly node: Node) {}

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
        if (this._externalInterceptor) this._externalInterceptor({ ...packet });
    }

    /**
     * Carga una biblioteca de funciones.
     *
     * @param library Biblioteca de funciones.
     */
    public loadLibrary(library: any | undefined): void {
        if (!library) {
            this._externalInterceptor = undefined;
            return;
        }

        const interceptor =
            Object.keys(library).find((e) => e === `intcp_${this.node.type}`) ??
            Object.keys(library).find((e) => e === "intcp");

        if (interceptor)
            this._externalInterceptor = (arg) =>
                library[interceptor](
                    {
                        mac: this.node.mac,
                        ip: this.node.ip,
                        name: this.node.name,
                        type: this.node.type as string,
                        send: (packet: Packet) => this.node.sendPacket(packet),
                    },
                    arg,
                );
    }
}
