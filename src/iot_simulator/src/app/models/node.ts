import { Packet } from "@models/packet";
import { Position } from "@models/position";

/**
 * Enum que representa los diferentes tipos de nodos en el simulador IoT.
 */
export enum NodeType {
    ROUTER = "router",
    COMPUTER = "computer",
    IOT = "iot",
}

/**
 * Representa un nodo dentro del esquema de red.
 */
export abstract class Node {
    /* Identificador único del nodo */
    private _mac: string;
    public get mac(): string {
        return this._mac;
    }
    /* Indentificador del nodo en la red */
    protected abstract _ip?: string;
    public get ip(): string | undefined {
        return this._ip;
    }
    /* Nombre del nodo */
    public name: string;
    /* Tipo de nodo */
    public type: NodeType;
    /* Indica si el nodo está comunicando */
    public abstract get communicating(): boolean;
    /* Tráfico del nodo */
    protected _traffic: Packet[];
    public get traffic(): Packet[] {
        return [...this._traffic];
    }
    /* Posición del nodo en el esquema de red */
    private _position: Position;
    public get position(): Position {
        return this._position;
    }

    /**
     * Crea una instancia de la clase Node.
     *
     * @param name Nombre del nodo.
     * @param type Tipo de nodo.
     * @param position Posición inicial del nodo.
     */
    public constructor(name: string, type: NodeType, position?: Position) {
        this._mac = this._generateMacAddress();
        this.name = name;
        this.type = type;
        this._traffic = [];
        this._position = position ?? { x: 0, y: 0 };
    }

    /**
     * Genera una dirección MAC aleatoria.
     *
     * @returns Dirección MAC aleatoria.
     */
    private _generateMacAddress(): string {
        return "XX:XX:XX:XX:XX:XX".replace(/X/g, () =>
            Math.floor(Math.random() * 16).toString(16),
        );
    }

    /**
     * Envía un paquete al nodo.
     *
     * @param packet Paquete a enviar.
     */
    public abstract sendPacket(packet: Packet): void;

    /**
     * Recibe un paquete del nodo.
     *
     * @param packet Paquete recibido.
     */
    public abstract receivePacket(packet: Packet): void;

    /**
     * Mueve el nodo en el esquema de red.
     *
     * @param x Desplazamiento horizontal.
     * @param y Desplazamiento vertical.
     */
    public move(x: number, y: number): void {
        this._position.x += x;
        this._position.y += y;
    }

    /**
     * Mueve el nodo a una posición concreta.
     *
     * @param x Coordenada x.
     * @param y Coordenada y.
     */
    public moveTo(x: number, y: number): void {
        this._position.x = x;
        this._position.y = y;
    }
}
