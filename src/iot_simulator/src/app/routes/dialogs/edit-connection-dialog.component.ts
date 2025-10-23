import { CommonModule } from "@angular/common";
import { Component } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmDialogImports } from "@components/ui/ui-dialog-helm/src";
import { HlmInputImports } from "@components/ui/ui-input-helm/src";
import { HlmLabelImports } from "@components/ui/ui-label-helm/src";
import { HlmSelectImports } from "@components/ui/ui-select-helm/src";
import { HlmSwitchImports } from "@components/ui/ui-switch-helm/src";
import { Connection } from "@models/connection";
import { TranslateModule } from "@ngx-translate/core";
import { injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectImports } from "@spartan-ng/brain/select";

export interface EditConnectionContext {
    connection: Connection;
}

@Component({
    selector: "app-edit-connection-dialog",
    imports: [
        CommonModule,
        FormsModule,
        TranslateModule,
        BrnSelectImports,
        HlmDialogImports,
        HlmInputImports,
        HlmLabelImports,
        HlmSelectImports,
        HlmSwitchImports,
    ],
    template: `
        <hlm-dialog-header>
            <h3 hlmDialogTitle>{{ "MODIFY_CONNECTION" | translate }}</h3>
        </hlm-dialog-header>
        <div class="grid gap-4">
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="justify-end">{{
                    "LATENCY" | translate
                }}</label>
                <input
                    hlmInput
                    type="number"
                    placeholder="ms"
                    [ngModel]="connection.latency"
                    (ngModelChange)="connection.latency = Math.max($event, 0)"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="justify-end">{{
                    "VARIABILITY" | translate
                }}</label>
                <input
                    hlmInput
                    type="number"
                    [ngModel]="connection.latencyVariation"
                    (ngModelChange)="
                        connection.latencyVariation = Math.max($event, 0)
                    "
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="justify-end">{{
                    "BANDWIDTH" | translate
                }}</label>
                <input
                    hlmInput
                    type="number"
                    [ngModel]="connection.bandwidth"
                    (ngModelChange)="connection.bandwidth = Math.max($event, 1)"
                    placeholder="Bytes/s"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="justify-end">{{
                    "CYBER_SHIELD" | translate
                }}</label>
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
                    <label hlmLabel class="justify-end">{{
                        "MODEL" | translate
                    }}</label>
                    <brn-select
                        [(ngModel)]="connection.cyberShield.model"
                        [disabled]="connection.cyberShield.models.length === 0"
                        [placeholder]="'SELECT_MODEL' | translate"
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
    host: { class: "min-w-96 flex flex-col gap-4" },
})
export class EditConnectionDialogComponent {
    public readonly Math: typeof Math = Math;
    protected readonly context =
        injectBrnDialogContext<EditConnectionContext>();
    protected readonly connection = this.context.connection;
}
