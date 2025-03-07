import { Connection, TransmittingStatus } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { Position } from "@models/position";
import { Router } from "@models/router";

/**
 * Clase que representa un dispositivo en la red.
 */
export class Device extends Node {
    /* Dirección IP del dispositivo */
    protected override _ip?: string | undefined;
    /* Conexión del dispositivo */
    private _connection?: Connection;
    /* Obtener si el dispositivo está comunicando */
    public override get communicating(): boolean {
        return (
            (this._connection?.transmitting ?? TransmittingStatus.NONE) !==
            TransmittingStatus.NONE
        );
    }

    /**
     * Constructor de la clase.
     *
     * @param name Nombre del dispositivo.
     * @param type Tipo del dispositivo.
     */
    constructor(name: string, type: NodeType, position?: Position) {
        if (type === NodeType.ROUTER)
            throw new Error("Device type cannot be a router");

        super(name, type, position);
    }

    /**
     * Conectar el dispositivo a un router.
     *
     * @param router Router al que se conectará el dispositivo.
     * @param latency Latencia de la conexión.
     */
    public connect(router: Router, latency: number = 0): void {
        [this._ip, this._connection] = router.acceptConnection(this, latency);
    }

    /**
     * Enviar un paquete al router al que está conectado el dispositivo.
     */
    public override sendPacket(packet: Packet): void {
        if (this._ip === undefined || this._connection === undefined)
            throw new Error("Device is not connected to a router");

        this._traffic.push(packet);
        this._connection.spreadPacket(packet);
    }

    /**
     * Recibir un paquete del router al que está conectado el dispositivo.
     */
    public override receivePacket(packet: Packet): void {
        if (packet.dstIP !== this._ip)
            console.error("Packet is not addressed to this device");

        this._traffic.push(packet);
        console.log("Packet received: ", packet);
    }
}
