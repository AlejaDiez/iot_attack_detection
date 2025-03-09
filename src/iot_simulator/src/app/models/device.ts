import { Connection, TransmittingStatus } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { Position } from "@models/position";
import { Router } from "@models/router";

/**
 * Clase que representa un dispositivo en la red.
 */
export class Device extends Node {
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
        if (this.ip === undefined || this._connection === undefined)
            throw new Error("Device is not connected to a router");

        this._traffic.push(packet);
        this._connection.spreadPacket(packet);
    }

    /**
     * Recibir un paquete del router al que está conectado el dispositivo.
     */
    public override receivePacket(packet: Packet): void {
        if (packet.dstIP !== this.ip)
            console.error("Packet is not addressed to this device");

        this._traffic.push(packet);
        console.log("Packet received: ", packet);
    }

    /**
     * Convierte un objeto plano a un router.
     *
     * @param obj Objeto plano que representa un router.
     * @returns Router creado a partir del objeto plano.
     */
    public static override fromObject(obj: any): Device {
        const device = new Device(
            obj.name,
            obj.type as NodeType,
            obj.position as Position,
        );

        if (obj.mac && typeof obj.mac === "string") device._mac = obj.mac;
        else throw new Error("MAC address is not valid");
        if (obj.ip && typeof obj.ip === "string") device._ip = obj.ip;
        else throw new Error("IP address is not valid");
        if (obj.traffic) {
            if (!Array.isArray(obj.traffic))
                throw new Error("Traffic is not valid");
            device._traffic = obj.traffic!.map((e: any) => e as Packet);
        }
        return device;
    }

    /**
     * Convertir el dispositivo a un objeto plano.
     *
     * @returns Objeto plano que representa el dispositivo.
     */
    public override toObject(): any {
        const traffic = this.traffic.map((e) => ({ ...e }));

        return {
            name: this.name,
            mac: this.mac,
            ip: this.ip,
            type: this.type,
            position: this.position,
            connection: this._connection?.toObject(),
            traffic: traffic.length ? traffic : undefined,
        };
    }
}
