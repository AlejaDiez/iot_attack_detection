import { Connection, Connections } from "./connection";
import { Position } from "./position";

/**
 * Enum que representa los diferentes tipos de dispositivos en el simulador IoT.
 */
export enum DeviceType {
    ROUTER = "router",
    COMPUTER = "computer",
    IOT = "iot",
}

/**
 * Representa un dispositivo en la red.
 */
export class Device {
    /** Dirección IP del nodo */
    private _ip: string;
    /** Tipo del nodo */
    private _type: DeviceType;
    /** Conexiones asociadas con el nodo */
    private _connections: Connections;
    /** Posición del nodo en la red */
    private _position: Position;
    /** Nombre del nodo */
    public name: string;

    /**
     * Getter de la propiedad ip.
     *
     * @returns Dirección IP del dispositivo
     */
    public get ip(): string {
        return this._ip;
    }

    /**
     * Getter de la propiedad type.
     *
     * @returns Tipo del dispositivo
     */
    public get type(): DeviceType {
        return this._type;
    }

    /**
     * Getter de la propiedad connections.
     *
     * @returns Conexiones asociadas con el dispositivo
     */
    public get connections(): Readonly<Connections> {
        return this._connections;
    }

    /**
     * Getter de la propiedad position.
     *
     * @returns Posición del dispositivo en la red
     */
    public get position(): Readonly<Position> {
        return this._position;
    }

    /**
     * Constructor de la clase dispositivo.
     *
     * @param ip - Dirección IP del dispositivo
     * @param name - Nombre del dispositivo
     * @param type - Tipo del dispositivo
     * @param connections - Conexiones asociadas con el dispositivo
     * @param position - Posición del dispositivo en la red
     */
    public constructor(
        ip: string,
        name: string,
        type: DeviceType = DeviceType.COMPUTER,
        connections: Connections = [],
        position: Position = { x: 0, y: 0 }
    ) {
        this._ip = ip;
        this.name = name;
        this._type = type;
        this._connections = connections;
        this._position = position;
    }

    /**
     * Envía datos a otro dispositivo.
     */
    public send(): void {
        // TODO
    }

    /**
     * Recibe datos de otro dispositivo.
     */
    public receive(): void {
        // TODO
    }

    /**
     * Conecta el dispositivo actual con otro dispositivo.
     *
     * @param device - Dispositivo con el que se desea conectar
     * @param latency - Latencia de la conexión
     * @returns Conexión entre los dos dispositivos
     */
    public connect(device: Device, latency: number): Connection {
        const connection: Connection = {
            origin: this,
            destination: device,
            latency,
        };

        this._connections.push(connection);
        device.acceptConnection(connection);
        return connection;
    }

    /**
     * Acepta una conexión con otro dispositivo.
     *
     * @param connection - Conexión a aceptar
     */
    public acceptConnection(connection: Connection): void {
        this._connections.push(connection);
    }

    /**
     * Movimiento del dispositivo en la red.
     *
     * @param position - Nueva posición del dispositivo
     */
    public move(position: Position): void {
        this._position = position;
    }
}
