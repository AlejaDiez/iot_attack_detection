import { Connection, TransmittingStatus } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { Position } from "@models/position";
import { Router } from "@models/router";

/** Tipo de dispositivo */
export type DeviceType = NodeType.COMPUTER | NodeType.IOT;

/**
 * Clase que representa un dispositivo en la red.
 */
export class Device extends Node {
    /* Conexión del dispositivo */
    private _connection?: Connection;
    public override get connected(): boolean {
        return this._connection !== undefined;
    }
    public override get communicating(): boolean {
        return (
            (this._connection?.transmitting ?? TransmittingStatus.NONE) !==
            TransmittingStatus.NONE
        );
    }

    /**
     * Constructor de la clase Device.
     *
     * @param name Nombre del dispositivo.
     * @param type Tipo del dispositivo.
     * @param position Posición inicial del dispositivo.
     */
    public constructor(name: string, type: DeviceType, position?: Position) {
        super(name, type, position);
    }

    /**
     * Conectar el dispositivo a un router.
     *
     * @param router Router al que se conectará el dispositivo.
     * @param latency Latencia de la conexión.
     */
    public connect(router: Router, latency?: number): void {
        [this.ip, this._connection] = router.acceptConnection(
            this,
            latency,
        ) ?? [undefined, undefined];

        if (this._connection === undefined)
            throw new Error(`${this.mac} could not connect to router`);
    }

    /**
     * Desconectar el dispositivo de un router.
     *
     * @param router Router al que se desconectará el dispositivo.
     */
    public disconnect(router: Router): void {
        const disconnected = router.removeConnection(this);

        if (disconnected) {
            this._connection = undefined;
            this.ip = undefined;
        }
    }

    public override sendPacket(packet: Packet): void {
        if (this._connection === undefined)
            throw new Error(`${this.mac} is not connected to a router`);

        this.logTraffic(packet);
        this._connection.spreadPacket(packet);
    }

    public override receivePacket(packet: Packet): void {
        if (this.ip !== packet.dstIP) {
            console.error(`Packet is not addressed to ${this.mac}`);
            return;
        }

        this.logTraffic(packet);
        console.log(
            `Packet received by ${this.mac}: ${JSON.stringify(packet, null, 2)}`,
        );
    }

    public static override fromObject(object: any): Device {
        const device = new Device(object.name, object.type, object.position);

        device.mac = object.mac;
        device.ip = object.ip;
        device.traffic = object.traffic;
        return device;
    }

    public override toObject() {
        const traffic = this.traffic;

        return {
            name: this.name,
            mac: this.mac,
            ip: this.ip,
            type: this.type,
            position: this.position,
            connection: this._connection?.toObject(),
            traffic: traffic.length > 0 ? traffic : undefined,
        };
    }
}
