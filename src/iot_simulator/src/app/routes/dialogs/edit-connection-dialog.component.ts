import { Component } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmSwitchModule } from "@components/ui/ui-switch-helm/src";
import { Connection } from "@models/connection";
import { injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import {
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,
} from "@spartan-ng/ui-dialog-helm";
import { HlmInputDirective } from "@spartan-ng/ui-input-helm";
import { HlmLabelDirective } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

export interface EditConnectionContext {
    connection: Connection;
}

@Component({
    selector: "app-edit-connection-dialog",
    imports: [
        FormsModule,
        BrnSelectModule,
        HlmDialogHeaderComponent,
        HlmDialogTitleDirective,
        HlmInputDirective,
        HlmLabelDirective,
        HlmSelectModule,
        HlmSwitchModule,
    ],
    template: `
        <hlm-dialog-header class="w-96">
            <h3 hlmDialogTitle>Modificar conexión</h3>
        </hlm-dialog-header>
        <div class="grid gap-4 py-4">
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Latencia</label>
                <input
                    hlmInput
                    type="number"
                    [(ngModel)]="connection.latency"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Variabilidad</label>
                <input
                    hlmInput
                    type="number"
                    [(ngModel)]="connection.latencyVariation"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Cyber Shield</label>
                <div class="col-span-3 inline-flex items-center justify-end">
                    <hlm-switch
                        [(ngModel)]="connection.cyberShield.enabled"
                        [disabled]="
                            connection.cyberShield.models.length === 0
                        " />
                </div>
            </div>
            @if (connection.cyberShield.enabled) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">Modelo</label>
                    <brn-select
                        placeholder="Selecciona un modelo"
                        [(ngModel)]="connection.cyberShield.model"
                        class="col-span-3 inline-block w-full">
                        <hlm-select-trigger class="w-full">
                            <hlm-select-value />
                        </hlm-select-trigger>
                        <hlm-select-content>
                            @for (
                                item of context.connection.cyberShield.models;
                                track $index
                            ) {
                                <hlm-option [value]="item.id">
                                    {{ item.name }}
                                </hlm-option>
                            }
                        </hlm-select-content>
                    </brn-select>
                </div>
            }
        </div>
    `,
    host: { class: "flex flex-col gap-4" },
})
export class EditConnectionDialogComponent {
    protected readonly context =
        injectBrnDialogContext<EditConnectionContext>();
    protected readonly connection = this.context.connection;
}
