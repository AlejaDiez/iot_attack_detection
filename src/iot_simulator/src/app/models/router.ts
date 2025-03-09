import { Connection } from "@models/connection";
import { Device } from "@models/device";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { Position } from "@models/position";

/* Tipo de la tabla ARP */
type ARP = [string, Connection];
/* Tabla ARP que contiene la dirección IP como clave, y la dirección MAC y conexión (interfaz) como valor */
type ARPTable = Map<string, ARP>;

/**
 * Clase que representa un router en la red.
 */
export class Router extends Node {
    /* Tabla ARP del router */
    private _arpTable: ARPTable = new Map<string, ARP>();
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
    constructor(name: string, position?: Position) {
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
        // Si el dispositivo no tiene ip, se le asigna una ip dinámica
        const ip = device.ip ?? `192.168.0.${this._arpTable.size + 2}`;
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
        if (packet.dstIP === this.ip) {
            this._traffic.push(packet);
            console.log("Paquete recibido por el router", packet);
        } else this._forwardPacket(packet);
    }

    /**
     * Convierte un objeto plano a un router.
     *
     * @param obj Objeto plano que representa un router.
     * @returns Router creado a partir del objeto plano.
     */
    public static override fromObject(obj: any): Router {
        const router = new Router(obj.name, obj.position as Position);

        if (obj.mac && typeof obj.mac === "string") router._mac = obj.mac;
        else throw new Error("MAC address is not valid");
        if (obj.ip && typeof obj.ip === "string") router._ip = obj.ip;
        else throw new Error("IP address is not valid");
        if (obj.traffic) {
            if (!Array.isArray(obj.traffic))
                throw new Error("Traffic is not valid");
            router._traffic = obj.traffic!.map((e: any) => e as Packet);
        }
        return router;
    }

    /**
     * Convierte el router a un objeto plano.
     *
     * @returns Objeto plano que representa el router.
     */
    public override toObject(): any {
        const traffic = this.traffic.map((e) => ({ ...e }));

        return {
            name: this.name,
            mac: this.mac.toString(),
            ip: this.ip,
            position: this.position,
            traffic: traffic.length ? traffic : undefined,
        };
    }
}
