import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { Connection } from "./connection";
import { Device } from "./device";

/**
 * Clase que representa un router en la red.
 */
export class Router extends Node {
    /* Identificador del nodo en la red */
    protected override _ip?: string | undefined;
    /* Tabla ARP que contiene la dirección IP como clave, y la dirección MAC y conexión (interfaz) como valor */
    private _arpTable: Map<string, [string, Connection]> = new Map<
        string,
        [string, Connection]
    >();
    /* Conexiones del router */
    public get connections(): Connection[] {
        return Array.from(this._arpTable.values()).map((value) => value[1]);
    }
    /* Indica si el nodo está comunicando */
    public override get communicating(): boolean {
        return this.connections.some((connection) => connection.transmitting);
    }

    /**
     * Crea una instancia de la clase Router.
     *
     * @param name Nombre del router.
     */
    constructor(name: string, position?: { x: number; y: number }) {
        super(name, NodeType.ROUTER, position);
        this._ip = "192.168.0.1";
        this._arpTable = new Map<string, [string, Connection]>();
    }

    /**
     * Reenvía un paquete a través de la red.
     *
     * @param packet Paquete a reenviar.
     */
    private _forwardPacket(packet: Packet): void {
        const [mac, connection] = this._arpTable.get(packet.dstIP) || [];

        this._traffic.push(packet);
        if (mac) connection!.spreadPacket(packet);
        else console.error("No se encontró la ip", packet.dstIP);
    }

    /**
     * Conectarse con un dispositivo.
     *
     * @param device Dispositivo que se conecta al router.
     * @returns Ip asignada al dispositivo y conexión establecida.
     */
    public acceptConnection(
        device: Device,
        latency: number,
    ): [string, Connection] {
        // Se crea una ip disponible para el dispositivo
        const ip = `192.168.0.${this._arpTable.size + 2}`;
        // Se crea una conexión entre el router y el dispositivo
        const connection = new Connection(this, device, latency);

        // Añadir la ip y la conexión a la tabla ARP
        this._arpTable.set(ip, [device.mac, connection]);

        // Asignar la ip al dispositivo
        return [ip, connection];
    }

    /**
     * Envía un paquete a través de la red.
     *
     * @param packet Paquete a enviar.
     */
    public override sendPacket(packet: Packet): void {
        this._forwardPacket(packet);
    }

    /**
     * Recibe un paquete de la red.
     *
     * @param packet Paquete recibido.
     */
    public override receivePacket(packet: Packet): void {
        if (packet.dstIP === this._ip) {
            this._traffic.push(packet);
            console.log("Paquete recibido por el router", packet);
        } else this._forwardPacket(packet);
    }
}
