import { Component } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmSwitchModule } from "@components/ui/ui-switch-helm/src";
import { Connection } from "@models/connection";
import { TranslateModule } from "@ngx-translate/core";
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
        TranslateModule,
    ],
    template: `
        <hlm-dialog-header>
            <h3 hlmDialogTitle>{{ "MODIFY_CONNECTION" | translate }}</h3>
        </hlm-dialog-header>
        <div class="grid gap-4 py-4">
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">{{
                    "LATENCY" | translate
                }}</label>
                <input
                    hlmInput
                    type="number"
                    [(ngModel)]="connection.latency"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">{{
                    "VARIABILITY" | translate
                }}</label>
                <input
                    hlmInput
                    type="number"
                    [(ngModel)]="connection.latencyVariation"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">{{
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
                    <label hlmLabel class="text-right">{{
                        "MODEL" | translate
                    }}</label>
                    <brn-select
                        [(ngModel)]="connection.cyberShield.model"
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
    protected readonly context =
        injectBrnDialogContext<EditConnectionContext>();
    protected readonly connection = this.context.connection;
}
