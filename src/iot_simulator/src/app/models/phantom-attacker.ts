import { Command, FlowGenerator } from "@models/flow-generator";
import { Packet } from "@models/packet";
import { paramsCount } from "@utils/parse_script";

/**
 * Ataque que se puede realizar.
 */
export interface Attack extends Command {}

/**
 * Ataques que se pueden realizar.
 */
export type Attacks = Attack[];

/**
 * Atacante Phantom.
 */
export class PhantomAttacker extends FlowGenerator {
    /** Ataques internos que se pueden realizar */
    public readonly internalAttacks: Attacks = [];
    /** Ataques externos que se pueden realizar */
    private _externalAttacks: Attacks = [];
    /** Ataques externos que se pueden realizar */
    public get externalAttacks(): Attacks {
        return [...this._externalAttacks];
    }

    public override loadLibrary(library: any | undefined): void {
        if (!library) {
            this.library = undefined;
            return;
        }

        const attacks = Object.keys(library).filter((e) =>
            e.startsWith("atk_"),
        );

        super.loadLibrary(library);
        this._externalAttacks = [];
        for (const atk of attacks) {
            this.library![atk] = (...args) =>
                library[atk](
                    {
                        mac: this.node.mac,
                        ip: this.node.ip,
                        name: this.node.name,
                        type: this.node.type as string,
                        send: (packet: Packet) => this.node.sendPacket(packet),
                    },
                    ...args,
                );
            this._externalAttacks.push({
                id: atk,
                name: atk.replace("atk_", ""),
                multiple: paramsCount(library[atk]) === Infinity,
            });
        }
    }

    /**
     * Ejecuta un ataque de una biblioteca.
     *
     * @param fn Nombre de la función a ejecutar.
     * @param args Argumentos de la función.
     */
    public attack(fn: string, ...args: any[]): void {
        if (this.library && this.library[fn]) this.library[fn](...args);
    }
}
