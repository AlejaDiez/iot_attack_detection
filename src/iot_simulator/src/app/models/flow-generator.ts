import { Node } from "@models/node";
import { paramsCount } from "@utils/parse_script";
import { Packet } from "./packet";

/**
 * Comando que se puede realizar.
 */
export interface Command {
    readonly id: string;
    readonly name: string;
    readonly multiple: boolean;
}

/**
 * Comandos que se pueden realizar.
 */
export type Commands = Command[];

/**
 * Generador de flujos de red.
 */
export class FlowGenerator {
    /** Biblioteca de funciones */
    protected library?: {
        [fn: string]: (...args: any[]) => void;
    };
    /** Comandos internos que se pueden realizar */
    public readonly internalCommands: Commands = [
        {
            id: "ping",
            name: "Ping",
            multiple: false,
        },
    ];
    /** Comandos externos que se pueden realizar */
    private _externalCommands: Commands = [];
    /** Comandos que se pueden realizar */
    public get externalCommands(): Commands {
        return [...this._externalCommands];
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
    public loadLibrary(library: any | undefined): void {
        if (!library) {
            this.library = undefined;
            return;
        }

        const commands = Object.keys(library).filter((e) =>
            e.startsWith("cmd_"),
        );

        this.library = {};
        this._externalCommands = [];
        for (const cmd of commands) {
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
            this._externalCommands.push({
                id: cmd,
                name: cmd.substring(4),
                multiple: paramsCount(library[cmd]) === Infinity,
            });
        }
    }

    /**
     * Ejecuta una función de la biblioteca.
     *
     * @param fn Nombre de la función a ejecutar.
     * @param args Argumentos de la función.
     */
    public execute(fn: string, ...args: any[]): void {
        if (this.library && this.library[fn]) this.library[fn](...args);
    }
}
