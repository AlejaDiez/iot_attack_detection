import { Node } from "@models/node";
import { Packet } from "./packet";

/**
 * Generador de flujos de red.
 */
export class FlowGenerator {
    /**
     * Crea un generador de flujos de red.
     *
     * @param node Nodo de red.
     */
    public constructor(private readonly node: Node) {}

    /**
     * Realiza un ping a una dirección IP.
     *
     * @param dstIP Dirección IP de destino.
     */
    public ping(dstIP: string): void {
        this.node.sendPacket(Packet.ICMPEchoRequest(this.node.ip ?? "", dstIP));
    }
}
