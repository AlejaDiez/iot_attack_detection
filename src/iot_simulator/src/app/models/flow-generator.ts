import { Node } from "@models/node";
import { Packet } from "./packet";

/**
 * Generador de flujos de red.
 */
export class FlowGenerator {
    /** Biblioteca de funciones */
    protected library?: {
        [fn: string]: (...args: any[]) => void;
    };
    /** Comandos que puede realizar el atacante */
    public get commands(): string[] {
        return Object.keys(this.library ?? {})
            .filter((e) => e.startsWith("cmd_"))
            .map((e) => e.substring(4));
    }

    /**
     * Crea un generador de flujos de red.
     *
     * @param node Nodo de red.
     */
    public constructor(protected readonly node: Node) {}

    /**
     * Realiza un ping a una dirección IP.
     *
     * @param dstIP Dirección IP de destino.
     */
    public ping(dstIP: string): void {
        this.node.sendPacket(Packet.ICMPEchoRequest(this.node.ip ?? "", dstIP));
    }

    /**
     * Carga una biblioteca de funciones.
     *
     * @param library Biblioteca de funciones.
     */
    public loadLibrary(library: any): void {
        const commands = Object.keys(library).filter((e) =>
            e.startsWith("cmd_"),
        );

        this.library = {};
        for (const cmd of commands)
            this.library[cmd] = (...args) =>
                library[cmd](
                    {
                        mac: this.node.mac,
                        ip: this.node.ip,
                        name: this.node.name,
                        type: this.node.type as string,
                        send: (packet: Packet) => this.node.sendPacket(packet),
                    },
                    ...args,
                );
    }

    /**
     * Ejecuta una función de la biblioteca.
     *
     * @param fn Nombre de la función a ejecutar.
     * @param args Argumentos de la función.
     */
    public execute(fn: string, ...args: any[]): void {
        if (this.library && this.library[`cmd_${fn}`])
            this.library[`cmd_${fn}`](...args);
    }
}
